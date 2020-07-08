#if !NPY_SIMD
    #error "Not a standalone header, only works through 'simd.dispatch.c.src'"
#endif

/************************************
 ** Protected Definitions
 ************************************/
static int
simd_arg_from_obj(PyObject *obj, simd_arg *arg)
{
    assert(arg->dtype != 0);
    const simd_data_info *info = simd_data_getinfo(arg->dtype);
    if (info->is_scalar) {
        arg->data = simd_scalar_from_obj(obj, arg->dtype);
    }
    else if (info->is_sequence) {
        unsigned min_seq_size = simd_data_getinfo(info->to_vector)->nlanes;
        arg->data.qu8 = simd_sequence_from_obj(obj, arg->dtype, min_seq_size);
    }
    else if (info->is_vectorx) {
        arg->data = simd_vectorx_from_obj(obj, arg->dtype);
    }
    else if (info->is_vector) {
        arg->data = simd_vector_from_obj((simd_vector*)obj, arg->dtype);
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
        return simd_scalar_to_obj(arg->data, arg->dtype);
    }
    if (info->is_sequence) {
        return simd_sequence_to_obj(arg->data.qu8, arg->dtype);
    }
    if (info->is_vectorx) {
        return simd_vectorx_to_obj(arg->data, arg->dtype);
    }
    if (info->is_vector) {
        return (PyObject*)simd_vector_to_obj(arg->data, arg->dtype);
    }
    PyErr_Format(PyExc_RuntimeError,
        "unhandled arg to object type id:%d, name:%s", arg->dtype, info->pyname
    );
    return NULL;
}

static void
simd_args_sequence_free(simd_arg *args, int args_len)
{
    assert(args_len > 0);
    while (--args_len >= 0) {
        simd_arg *arg = &args[args_len];
        const simd_data_info *info = simd_data_getinfo(arg->dtype);
        if (!info->is_sequence) {
            continue;
        }
        simd_sequence_free(arg->data.qu8);
    }
}

static int
simd_args_from_tuple(PyObject *tuple_obj, simd_arg *args, int args_len, const char *method_name)
{
    assert(args_len > 0);
    assert(PyTuple_Check(tuple_obj));

    Py_ssize_t obj_arg_len = PyTuple_GET_SIZE(tuple_obj);
    if (obj_arg_len != args_len) {
        if (args_len == 1) {
            PyErr_Format(PyExc_TypeError,
                "%s() takes only one argument (%d given)", method_name, obj_arg_len
            );
            return -1;
        }
        PyErr_Format(PyExc_TypeError,
            "%s() takes exactly %d arguments (%d given)", method_name, args_len, obj_arg_len
        );
        return -1;
    }
    for (int arg_pos = 0; arg_pos < args_len; ++arg_pos) {
        simd_arg *arg = &args[arg_pos];
        arg->obj = PyTuple_GET_ITEM(tuple_obj, arg_pos);
        assert(arg->obj != NULL);
        if (simd_arg_from_obj(arg->obj, arg) != 0) {
            // free previous args
            if (arg_pos > 0) {
                simd_args_sequence_free(args, arg_pos);
            }
            // TODO: improve log by add argument number and method name
            return -1;
        }
    }
    return 0;
}
