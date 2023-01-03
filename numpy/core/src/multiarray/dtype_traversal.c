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

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"

#include "alloc.h"
#include "array_method.h"
#include "dtypemeta.h"

#include "dtype_traversal.h"


/* Same as in dtype_transfer.c */
#define NPY_LOWLEVEL_BUFFER_BLOCKSIZE  128


/*
 * Generic Clear function helpers:
 */

static int
get_clear_function(
        void *traverse_context, PyArray_Descr *dtype, int aligned,
        npy_intp stride, NPY_traverse_info *clear_info,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    NPY_traverse_info_init(clear_info);

    get_simple_loop_function *get_clear = NPY_DT_SLOTS(NPY_DTYPE(dtype))->get_clear_loop;
    if (get_clear == NULL) {
        PyErr_Format(PyExc_RuntimeError,
                "Internal error, tried to fetch decref/clear function for the "
                "unsupported DType '%S'.", dtype);
        return -1;
    }

    if (get_clear(traverse_context, dtype, aligned, stride,
                  &clear_info->func, &clear_info->auxdata, flags) <  0) {
        /* callee should clean up, but make sure outside debug mode */
        assert(clear_info->func == NULL);
        clear_info->func = NULL;
        return -1;
    }
    Py_INCREF(dtype);
    clear_info->descr = dtype;

    return 0;
}

/*
 * Helper to set up a strided loop used for clearing.
 * The function will error when called on a dtype which does not have
 * references (and thus the get_clear_loop slot NULL).
 * Note that old-style user-dtypes use the "void" version.
 *
 * NOTE: This function may have a use for a `traverse_context` at some point
 *       but right now, it is always NULL and only exists to allow adding it
 *       in the future without changing the strided-loop signature.
 */
NPY_NO_EXPORT int
PyArray_GetClearFunction(
        int aligned, npy_intp stride, PyArray_Descr *dtype,
        NPY_traverse_info *clear_info, NPY_ARRAYMETHOD_FLAGS *flags)
{
    return get_clear_function(NULL, dtype, aligned, stride, clear_info, flags);
}


/****************** Python Object clear ***********************/

static int
clear_object_strided_loop(
        void *NPY_UNUSED(traverse_context), PyArray_Descr *NPY_UNUSED(descr),
        char *data, npy_intp size, npy_intp stride,
        NpyAuxData *NPY_UNUSED(auxdata))
{
    PyObject *aligned_copy = NULL;
    while (size > 0) {
        /* Release the reference in src and set it to NULL */
        memcpy(&aligned_copy, data, sizeof(PyObject *));
        Py_XDECREF(aligned_copy);
        memset(data, 0, sizeof(PyObject *));

        data += stride;
        --size;
    }
    return 0;
}


NPY_NO_EXPORT int
npy_get_clear_object_strided_loop(
        void *NPY_UNUSED(traverse_context), PyArray_Descr *NPY_UNUSED(descr),
        int NPY_UNUSED(aligned), npy_intp NPY_UNUSED(fixed_stride),
        simple_loop_function **out_loop, NpyAuxData **out_auxdata,
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
        NPY_traverse_info_xfree(&d->fields[i].info);
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
traverse_fields_function(
        void *traverse_context, PyArray_Descr *NPY_UNUSED(descr),
        char *data, npy_intp N, npy_intp stride,
        NpyAuxData *auxdata)
{
    fields_clear_data *d = (fields_clear_data *)auxdata;
    npy_intp i, field_count = d->field_count;

    /* Do the traversing a block at a time for better memory caching */
    const npy_intp blocksize = NPY_LOWLEVEL_BUFFER_BLOCKSIZE;

    for (;;) {
        if (N > blocksize) {
            for (i = 0; i < field_count; ++i) {
                single_field_clear_data field = d->fields[i];
                if (field.info.func(traverse_context,
                        field.info.descr, data + field.src_offset,
                        blocksize, stride, field.info.auxdata) < 0) {
                    return -1;
                }
            }
            N -= blocksize;
            data += blocksize * stride;
        }
        else {
            for (i = 0; i < field_count; ++i) {
                single_field_clear_data field = d->fields[i];
                if (field.info.func(traverse_context,
                        field.info.descr, data + field.src_offset,
                        N, stride, field.info.auxdata) < 0) {
                    return -1;
                }
            }
            return 0;
        }
    }
}


static int
get_clear_fields_transfer_function(
        void *traverse_context, PyArray_Descr *dtype, int NPY_UNUSED(aligned),
        npy_intp stride, simple_loop_function **out_func,
        NpyAuxData **out_auxdata, NPY_ARRAYMETHOD_FLAGS *flags)
{
    PyObject *names, *key, *tup, *title;
    PyArray_Descr *fld_dtype;
    npy_int i, structsize;
    Py_ssize_t field_count;

    names = dtype->names;
    field_count = PyTuple_GET_SIZE(dtype->names);

    /* Over-allocating here: less fields may be used */
    structsize = (sizeof(fields_clear_data) +
                    field_count * sizeof(single_field_clear_data));
    /* Allocate the data and populate it */
    fields_clear_data *data = PyMem_Malloc(structsize);
    if (data == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    data->base.free = &fields_clear_data_free;
    data->base.clone = &fields_clear_data_clone;
    data->field_count = 0;

    single_field_clear_data *field = data->fields;
    for (i = 0; i < field_count; ++i) {
        int offset;

        key = PyTuple_GET_ITEM(names, i);
        tup = PyDict_GetItem(dtype->fields, key);
        if (!PyArg_ParseTuple(tup, "Oi|O", &fld_dtype, &offset, &title)) {
            NPY_AUXDATA_FREE((NpyAuxData *)data);
            return -1;
        }
        if (PyDataType_REFCHK(fld_dtype)) {
            NPY_ARRAYMETHOD_FLAGS clear_flags;
            if (get_clear_function(
                    traverse_context, fld_dtype, 0,
                    stride, &field->info, &clear_flags) < 0) {
                NPY_AUXDATA_FREE((NpyAuxData *)data);
                return -1;
            }
            *flags = PyArrayMethod_COMBINED_FLAGS(*flags, clear_flags);
            field->src_offset = offset;
            data->field_count++;
            field++;
        }
    }

    *out_func = &traverse_fields_function;
    *out_auxdata = (NpyAuxData *)data;

    return 0;
}


typedef struct {
    NpyAuxData base;
    npy_intp count;
    NPY_traverse_info info;
} subarray_clear_data;


/* transfer data free function */
static void
subarray_clear_data_free(NpyAuxData *data)
{
    subarray_clear_data *d = (subarray_clear_data *)data;

    NPY_traverse_info_xfree(&d->info);
    PyMem_Free(d);
}


/* transfer data copy function */
static NpyAuxData *
subarray_clear_data_clone(NpyAuxData *data)
{
    subarray_clear_data *d = (subarray_clear_data *)data;

    /* Allocate the data and populate it */
    subarray_clear_data *newdata = PyMem_Malloc(sizeof(subarray_clear_data));
    if (newdata == NULL) {
        return NULL;
    }
    newdata->base = d->base;
    newdata->count = d->count;

    if (NPY_traverse_info_copy(&newdata->info, &d->info) < 0) {
        PyMem_Free(newdata);
        return NULL;
    }

    return (NpyAuxData *)newdata;
}


static int
traverse_subarray_func(
        void *traverse_context, PyArray_Descr *NPY_UNUSED(descr),
        char *data, npy_intp N, npy_intp stride,
        NpyAuxData *auxdata)
{
    subarray_clear_data *subarr_data = (subarray_clear_data *)auxdata;

    simple_loop_function *func = subarr_data->info.func;
    PyArray_Descr *sub_descr = subarr_data->info.descr;
    npy_intp sub_N = subarr_data->count;
    NpyAuxData *sub_auxdata = subarr_data->info.auxdata;
    npy_intp sub_stride = sub_descr->elsize;

    while (N--) {
        if (func(traverse_context, sub_descr, data,
                 sub_N, sub_stride, sub_auxdata) < 0) {
        return -1;
        }
        data += stride;
    }
    return 0;
}


static int
get_subarray_clear_func(
        void *traverse_context, PyArray_Descr *dtype, int aligned,
        npy_intp size, npy_intp stride, simple_loop_function **out_func,
        NpyAuxData **out_auxdata, NPY_ARRAYMETHOD_FLAGS *flags)
{
    subarray_clear_data *auxdata = PyMem_Malloc(sizeof(subarray_clear_data));
    if (auxdata == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    auxdata->count = size;
    auxdata->base.free = &subarray_clear_data_free;
    auxdata->base.clone = &subarray_clear_data_clone;

    if (get_clear_function(
            traverse_context, dtype, aligned,
            dtype->elsize, &auxdata->info, flags) < 0) {
        PyMem_Free(auxdata);
        return -1;
    }
    *out_func = &traverse_subarray_func;
    *out_auxdata = (NpyAuxData *)auxdata;

    return 0;
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
        void *traverse_context, PyArray_Descr *dtype, int aligned,
        npy_intp stride, simple_loop_function **out_func,
        NpyAuxData **out_auxdata, NPY_ARRAYMETHOD_FLAGS *flags)
{
    /*
     * If there are no references, it's a nop.  This path should not be hit
     * but structured dtypes are tricky when a dtype which included references
     * was sliced to not include any.
     */
    if (!PyDataType_REFCHK(dtype)) {
        *out_func = &clear_no_op;
        return 0;
    }

    if (PyDataType_HASSUBARRAY(dtype)) {
        PyArray_Dims shape = {NULL, -1};
        npy_intp size;

        if (!(PyArray_IntpConverter(dtype->subarray->shape, &shape))) {
            PyErr_SetString(PyExc_ValueError,
                    "invalid subarray shape");
            return -1;
        }
        size = PyArray_MultiplyList(shape.ptr, shape.len);
        npy_free_cache_dim_obj(shape);

        if (get_subarray_clear_func(
                traverse_context, dtype->subarray->base, aligned, size, stride,
                out_func, out_auxdata, flags) < 0) {
            return -1;
        }

        return 0;
    }
    /* If there are fields, need to do each field */
    else if (PyDataType_HASFIELDS(dtype)) {
        if (get_clear_fields_transfer_function(
                traverse_context, dtype, aligned, stride,
                out_func, out_auxdata, flags) < 0) {
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
