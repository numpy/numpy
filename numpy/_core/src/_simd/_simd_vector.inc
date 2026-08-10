/**
 * This file is included by `_simd.dispatch.c.src`. Its contents are affected by the simd configuration, and
 * therefore must be built multiple times. Making it a standalone `.c` file with `NPY_VISIBILITY_HIDDEN`
 * symbols would require judicious use of `NPY_CPU_DISPATCH_DECLARE` and `NPY_CPU_DISPATCH_CURFX`, which was
 * deemed too harmful to readability.
 */
/************************************
 ** Private Definitions
 ************************************/
static Py_ssize_t
simd__vector_length(PySIMDVectorObject *self)
{
    return simd_data_getinfo(self->dtype)->nlanes;
}
static PyObject *
simd__vector_item(PySIMDVectorObject *self, Py_ssize_t i)
{
    const simd_data_info *info = simd_data_getinfo(self->dtype);
    int nlanes = info->nlanes;
    if (i >= nlanes) {
        PyErr_SetString(PyExc_IndexError, "vector index out of range");
        return NULL;
    }
    npyv_lanetype_u8 *src = self->data + i * info->lane_size;
    simd_data data;
    memcpy(&data.u64, src, info->lane_size);
    return simd_scalar_to_number(data, info->to_scalar);
}

static PyObject *
simd__vector_name(PySIMDVectorObject *self, void *NPY_UNUSED(ignored))
{
    return PyUnicode_FromString(simd_data_getinfo(self->dtype)->pyname);
}
static PyGetSetDef simd__vector_getset[] = {
    { "__name__", (getter)simd__vector_name, NULL, NULL, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

static PyObject *
simd__vector_repr(PySIMDVectorObject *self)
{
    PyObject *obj = PySequence_List((PyObject*)self);
    if (obj != NULL) {
        const char *type_name = simd_data_getinfo(self->dtype)->pyname;
        PyObject *repr = PyUnicode_FromFormat("<%s of %R>", type_name, obj);
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
// The type is the only reference a PySIMDVectorObject holds.
static int
simd__vector_traverse(PyObject *self, visitproc visit, void *arg)
{
    Py_VISIT(Py_TYPE(self));
    return 0;
}

static PyType_Slot simd__vector_slots[] = {
    {Py_tp_repr,        (void*)simd__vector_repr},
    {Py_tp_richcompare, (void*)simd__vector_compare},
    {Py_tp_getset,      (void*)simd__vector_getset},
    {Py_sq_length,      (void*)simd__vector_length},
    {Py_sq_item,        (void*)simd__vector_item},
    {Py_tp_traverse,    (void*)simd__vector_traverse},
    {0, NULL}
};

static PyType_Spec simd__vector_spec = {
    /*
     * The dot is required: it splits `__module__` from `__name__`, which stays
     * VECTOR_<TARGET>. Without one `__module__` is undefined, which CPython
     * reports as a DeprecationWarning.
     */
    .name = SIMD_MODULE_NAME "." NPY_TOSTRING(NPY_CPU_DISPATCH_CURFX(VECTOR)),
    .basicsize = sizeof(PySIMDVectorObject),
    /*
     * Heap types are mutable and instantiable by default; the last two flags
     * restore the static type's behaviour. HAVE_GC is required because
     * instances reference their type.
     */
    .flags = Py_TPFLAGS_DEFAULT
             | Py_TPFLAGS_HAVE_GC
             | Py_TPFLAGS_IMMUTABLETYPE
             | Py_TPFLAGS_DISALLOW_INSTANTIATION,
    .slots = simd__vector_slots
};

/*
 * One per compiled dispatch target, like the static PyTypeObject it replaces.
 * Module state would need multi-phase init and a module handle at every
 * intrinsic call site.
 */
static PyObject *PySIMDVectorType = NULL;

/************************************
 ** Protected Definitions
 ************************************/
/*
 * Force inlining the following functions on CYGWIN to avoid spilling vector
 * registers into the stack to workaround GCC/WIN64 bug that performs
 * miss-align load variable of 256/512-bit vector from non-aligned
 * 256/512-bit stack pointer.
 *
 * check the following links for more clarification:
 * https://github.com/numpy/numpy/pull/18330#issuecomment-821539919
 * https://gcc.gnu.org/bugzilla/show_bug.cgi?id=49001
 */
#if defined(__CYGWIN__) || (defined(__GNUC__) && defined(_WIN64))
    #define CYG_FINLINE NPY_FINLINE
#else
    #define CYG_FINLINE static
#endif
CYG_FINLINE PySIMDVectorObject *
PySIMDVector_FromData(simd_data data, simd_data_type dtype)
{
    const simd_data_info *info = simd_data_getinfo(dtype);
    assert(info->is_vector && info->nlanes > 0);

    /*
     * PyObject_New() must not be used with Py_TPFLAGS_HAVE_GC, and skips the
     * reference on the type that a heap type's instance owns.
     * PyType_GenericAlloc() is this type's tp_alloc, which the Limited API
     * cannot reach directly; it increfs the type and GC-tracks the object.
     */
    PySIMDVectorObject *vec = (PySIMDVectorObject *)PyType_GenericAlloc(
        (PyTypeObject *)PySIMDVectorType, 0
    );
    if (vec == NULL) {
        return NULL;
    }
    vec->dtype = dtype;
    if (info->is_bool) {
        // boolean vectors are internally treated as unsigned
        // vectors to add compatibility among all SIMD extensions
        switch(dtype) {
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

CYG_FINLINE simd_data
PySIMDVector_AsData(PySIMDVectorObject *vec, simd_data_type dtype)
{
    const simd_data_info *info = simd_data_getinfo(dtype);
    assert(info->is_vector && info->nlanes > 0);

    simd_data data = {.u64 = 0};
    if (!PyObject_IsInstance((PyObject *)vec, PySIMDVectorType)) {
        PyErr_Format(PyExc_TypeError,
            "a vector type %s is required", info->pyname
        );
        return data;
    }
    if (vec->dtype != dtype) {
        PyErr_Format(PyExc_TypeError,
            "a vector type %s is required, got(%s)",
            info->pyname, simd_data_getinfo(vec->dtype)->pyname
        );
        return data;
    }

    data.vu8 = npyv_load_u8(vec->data);
    if (info->is_bool) {
        // boolean vectors are internally treated as unsigned
        // vectors to add compatibility among all SIMD extensions
        switch(dtype) {
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
PySIMDVectorType_Init(PyObject *module)
{
    PySIMDVectorType = PyType_FromSpec(&simd__vector_spec);
    if (PySIMDVectorType == NULL) {
        return -1;
    }
    return PyModule_AddObjectRef(module, "vector_type", PySIMDVectorType);
}
