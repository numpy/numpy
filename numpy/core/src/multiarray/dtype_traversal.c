/*
 * This file is simlar to the low-level loops for data type transfer
 * in `dtype_transfer.c` but for those which only require visiting
 * a single array (and mutating it in-place).
 *
 * As of writing, it is only used for CLEARing, which means mainly
 * Python object DECREF/dealloc followed by NULL'ing the data
 * (to support double clearing on errors).
 * However, memory initialization and traverse follows similar
 * protocols (although traversal needs additional arguments.
 */


#include "dtypemeta.h"

#include "dtype_traversal.h"


/****************** Python Object clear ***********************/


static int
clear_object_strided_loop(
        void *NPY_UNUSED(traverse_context), PyArray_Descr *NPY_UNUSED(descr),
        char *data, npy_intp size, npy_intp stride,
        NpyAuxData *NPY_UNUSED(auxdata))
{
    PyObject *alignd_copy = NULL;
    while (size > 0) {
        /* Release the reference in src and set it to NULL */
        memcpy(&alignd_copy, data, sizeof(PyObject *));
        Py_XDECREF(alignd_copy);
        memset(data, 0, sizeof(PyObject *));

        data += stride;
        --size;
    }
    return 0;
}


NPY_NO_EXPORT int
npy_get_clear_object_strided_loop(
        void *NPY_UNUSED(traverse_context), int NPY_UNUSED(aligned),
        npy_intp NPY_UNUSED(fixed_stride),
        simple_loop_function **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    *out_loop = &clear_object_strided_loop;
    return 0;
}


/**************** Structured DType clear funcationality ***************/

/*
 * Note that legacy user dtypes also make use of this.  Someone managed to
 * hack objects into them by adding a field that contains objects and this
 * remains (somewhat) valid.
 * (Unlike our voids, those fields must be hardcoded probably, but...)
 *
 * The below functionality mirrors the casting functionality relatively
 * closely.
 */

typedef struct {
    npy_intp src_offset;
    NPY_traverse_info info;
} single_field_clear_data;

typedef struct {
    NpyAuxData base;
    npy_intp field_count;
    single_field_clear_data fields[];
} fields_clear_data;


/* transfer data free function */
static void
fields_clear_data_free(NpyAuxData *data)
{
    fields_clear_data *d = (fields_clear_data *)data;

    for (npy_intp i = 0; i < d->field_count; ++i) {
        NPY_traverse_info_free(&d->fields[i].info);
    }
    PyMem_Free(d);
}


/* transfer data copy function */
static NpyAuxData *
fields_clear_data_clone(NpyAuxData *data)
{
    fields_clear_data *d = (fields_clear_data *)data;

    npy_intp field_count = d->field_count;
    npy_intp structsize = sizeof(fields_clear_data) +
                    field_count * sizeof(single_field_clear_data);

    /* Allocate the data and populate it */
    fields_clear_data *newdata = PyMem_Malloc(structsize);
    if (newdata == NULL) {
        return NULL;
    }
    newdata->base = d->base;
    newdata->field_count = 0;

    /* Copy all the fields transfer data */
    single_field_clear_data *in_field = d->fields;
    single_field_clear_data *new_field = newdata->fields;

    for (; newdata->field_count < field_count;
                newdata->field_count++, in_field++, new_field++) {
        new_field->src_offset = in_field->src_offset;

        if (NPY_traverse_info_copy(&new_field->info, &in_field->info) < 0) {
            fields_clear_data_free((NpyAuxData *)newdata);
            return NULL;
        }
    }

    return (NpyAuxData *)newdata;
}


static int
get_clear_fields_transfer_function(
        void *NPY_UNUSED(traverse_context), PyArray_Descr *dtype,
        npy_intp stride, simple_loop_function **out_func,
        NpyAuxData **out_transferdata, NPY_ARRAYMETHOD_FLAGS *flags)
{
    PyObject *names, *key, *tup, *title;
    PyArray_Descr *fld_dtype;
    npy_int i, structsize;
    Py_ssize_t field_count;
    int src_offset;

    names = dtype->names;
    field_count = PyTuple_GET_SIZE(dtype->names);

    /* Over-allocating here: less fields may be used */
    structsize = (sizeof(fields_clear_data) +
                    field_count * sizeof(single_field_clear_data));
    /* Allocate the data and populate it */
    fields_clear_data *data = PyMem_Malloc(structsize);
    if (data == NULL) {
        PyErr_NoMemory();
        return NPY_FAIL;
    }
    data->base.free = &fields_clear_data_free;
    data->base.clone = &fields_clear_data_clone;
    data->field_count = 0;

    single_field_clear_data *field = data->fields;
    for (i = 0; i < field_count; ++i) {
        key = PyTuple_GET_ITEM(names, i);
        tup = PyDict_GetItem(dtype->fields, key);
        if (!PyArg_ParseTuple(tup, "Oi|O", &fld_dtype,
                                                &offset, &title)) {
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return NPY_FAIL;
        }
        if (PyDataType_REFCHK(fld_dtype)) {
            NPY_ARRAYMETHOD_FLAGS clear_flags;
            if (get_clear_function(
                    0, stride, fld_dtype,
                    &field->info, &clear_flags) < 0) {
                NPY_AUXDATA_FREE((NpyAuxData *)data);
                return NPY_FAIL;
            }
            *flags = PyArrayMethod_COMBINED_FLAGS(*flags, clear_flags);
            field->src_offset = src_offset;
            data->field_count++;
            field++;
        }
    }

    *out_stransfer = &_strided_to_strided_field_transfer;
    *out_transferdata = (NpyAuxData *)data;

    return NPY_SUCCEED;
}


static int
clear_no_op(
        void *NPY_UNUSED(traverse_context), PyArray_Descr *NPY_UNUSED(descr),
        char *NPY_UNUSED(data), npy_intp NPY_UNUSED(size),
        npy_intp NPY_UNUSED(stride), NpyAuxData *NPY_UNUSED(auxdata))
{
    return 0;
}


NPY_NO_EXPORT int
npy_get_clear_void_and_legacy_user_dtype_loop(
        void *NPY_UNUSED(traverse_context), PyArray_Descr *dtype,
        npy_intp stride, simple_loop_function **out_func,
        NpyAuxData **out_transferdata, NPY_ARRAYMETHOD_FLAGS *flags)
{
    /* If there are no references, it's a nop (path should not be hit?) */
    if (!PyDataType_REFCHK(dtype)) {
        *out_loop = &clear_no_op;
        *out_transferdata = NULL;
        assert(0);
        return 0;
    }

    if (PyDataType_HASSUBARRAY(dtype)) {
        PyArray_Dims src_shape = {NULL, -1};
        npy_intp src_size;

        if (!(PyArray_IntpConverter(dtype->subarray->shape,
                                            &src_shape))) {
            PyErr_SetString(PyExc_ValueError,
                    "invalid subarray shape");
            return -1;
        }
        src_size = PyArray_MultiplyList(src_shape.ptr, src_shape.len);
        npy_free_cache_dim_obj(src_shape);

        if (get_n_to_n_transfer_function(aligned,
                stride, 0,
                dtype->subarray->base, NULL, 1, src_size,
                out_loop, out_transferdata,
                flags) != NPY_SUCCEED) {
            return -1;
        }

        return 0;
    }
    /* If there are fields, need to do each field */
    else if (PyDataType_HASFIELDS(dtype)) {
        if (get_clear_fields_transfer_function(
                aligned, stride, dtype,
                out_loop, out_transferdata, flags) < 0) {
            return -1;
        }
        return 0;
    }

    PyErr_Format(PyExc_RuntimeError,
            "Internal error, tried to fetch clear function for the "
            "user dtype '%s' without fields or subarray (legacy support).",
            dtype);
    return -1;
}
