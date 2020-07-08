#if !NPY_SIMD
    #error "Not a standalone header, only works through 'simd.dispatch.c.src'"
#endif

/************************************
 ** Private Definitions
 ************************************/
// PySequenceMethods
static Py_ssize_t
simd__vector_length(simd_vector *self)
{
    return simd_data_getinfo(self->type)->nlanes;
}
static PyObject *
simd__vector_item(simd_vector *self, Py_ssize_t i)
{
    const simd_data_info *info = simd_data_getinfo(self->type);
    int nlanes = info->nlanes;
    if (i >= nlanes) {
        PyErr_SetString(PyExc_IndexError, "list index out of range");
        return NULL;
    }
    npyv_lanetype_u8 *src = self->data + i * info->lane_size;
    simd_data data;
    memcpy(&data.u64, src, info->lane_size);
    return simd_scalar_to_obj(data, info->to_scalar);
}

static PySequenceMethods simd__vector_as_sequence = {
    (lenfunc) simd__vector_length,           /* sq_length */
    (binaryfunc) NULL,                       /* sq_concat */
    (ssizeargfunc) NULL,                     /* sq_repeat */
    (ssizeargfunc) simd__vector_item,        /* sq_item */
    (ssizessizeargfunc) NULL,                /* sq_slice */
    (ssizeobjargproc) NULL,                  /* sq_ass_item */
    (ssizessizeobjargproc) NULL,             /* sq_ass_slice */
    (objobjproc) NULL,                       /* sq_contains */
    (binaryfunc) NULL,                       /* sq_inplace_concat */
    (ssizeargfunc) NULL,                     /* sq_inplace_repeat */
};

// PyGetSetDef
static PyObject *
simd__vector_name(simd_vector *self)
{
    return PyUnicode_FromString(simd_data_getinfo(self->type)->pyname);
}
static PyGetSetDef simd__vector_getset[] = {
    { "__name__", (getter)simd__vector_name, NULL, NULL, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

// PyTypeObject(simd__vector_type)
static PyObject *
simd__vector_repr(PyObject *self)
{
    // PySequence_Fast returns Tuple in PyPy
    PyObject *obj = PySequence_List(self);
    if (obj != NULL) {
        PyObject *repr = PyObject_Str(obj);
        Py_DECREF(obj);
        return repr;
    }
    return obj;
}
static PyObject *
simd__vector_compare(PyObject *self, PyObject *other, int cmp_op)
{
    PyObject *obj;
    if (PyTuple_Check(other)) {
        obj = PySequence_Tuple(self);
    } else if (PyList_Check(other)) {
        obj = PySequence_List(self);
    } else {
        obj = PySequence_Fast(self, "invalid argument, expected a vector");
    }
    if (obj != NULL) {
        PyObject *rich = PyObject_RichCompare(obj, other, cmp_op);
        Py_DECREF(obj);
        return rich;
    }
    return obj;
}
static PyTypeObject simd__vector_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = NPY_TOSTRING(NPY_CPU_DISPATCH_CURFX(VECTOR)),
    .tp_basicsize = sizeof(simd_vector),
    .tp_repr = (reprfunc)simd__vector_repr,
    .tp_as_sequence = &simd__vector_as_sequence,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_richcompare = simd__vector_compare,
    .tp_getset = simd__vector_getset
};

/************************************
 ** Protected Definitions
 ************************************/
static simd_vector *
simd_vector_to_obj(simd_data data, simd_data_type vtype)
{
    const simd_data_info *info = simd_data_getinfo(vtype);
    assert(info->is_vector && info->nlanes > 0);

    simd_vector *vec = PyObject_New(simd_vector, &simd__vector_type);
    if (vec == NULL) {
        return (simd_vector*)PyErr_NoMemory();
    }
    vec->type = vtype;
    if (info->is_bool) {
        // boolean vectors are internally treated as unsigned
        // vectors to add compatibility among all SIMD extensions
        switch(vtype) {
        case simd_data_vb8:
            data.vu8 = npyv_cvt_u8_b8(data.vb8);
            break;
        case simd_data_vb16:
            data.vu16 = npyv_cvt_u16_b16(data.vb16);
            break;
        case simd_data_vb32:
            data.vu32 = npyv_cvt_u32_b32(data.vb32);
            break;
        default:
            data.vu64 = npyv_cvt_u64_b64(data.vb64);
        }
    }
    npyv_store_u8(vec->data, data.vu8);
    return vec;
}

static simd_data
simd_vector_from_obj(simd_vector *vec, simd_data_type vtype)
{
    const simd_data_info *info = simd_data_getinfo(vtype);
    assert(info->is_vector && info->nlanes > 0);

    simd_data data = {.u64 = 0};
    if (!PyObject_IsInstance(
        (PyObject *)vec, (PyObject *)&simd__vector_type
    )) {
        PyErr_Format(PyExc_TypeError,
            "a vector type %s is required", info->pyname
        );
        return data;
    }
    if (vec->type != vtype) {
        PyErr_Format(PyExc_TypeError,
            "a vector type %s is required, got(%s)",
            info->pyname, simd_data_getinfo(vec->type)->pyname
        );
        return data;
    }

    data.vu8 = npyv_load_u8(vec->data);
    if (info->is_bool) {
        // boolean vectors are internally treated as unsigned
        // vectors to add compatibility among all SIMD extensions
        switch(vtype) {
        case simd_data_vb8:
            data.vb8 = npyv_cvt_b8_u8(data.vu8);
            break;
        case simd_data_vb16:
            data.vb16 = npyv_cvt_b16_u16(data.vu16);
            break;
        case simd_data_vb32:
            data.vb32 = npyv_cvt_b32_u32(data.vu32);
            break;
        default:
            data.vb64 = npyv_cvt_b64_u64(data.vu64);
        }
    }
    return data;
}

static int
simd_vector_register(PyObject *module)
{
    Py_INCREF(&simd__vector_type);
    if (PyType_Ready(&simd__vector_type)) {
        return -1;
    }
    if (PyModule_AddObject(
        module, "vector_type",(PyObject *)&simd__vector_type
    )) {
        return -1;
    }
    return 0;
}
