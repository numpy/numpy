#include "field_types.h"
#include "conversions.h"
#include "str_to_int.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/ndarraytypes.h"
#include "alloc.h"

#include "textreading/growth.h"


NPY_NO_EXPORT void
field_types_xclear(int num_field_types, field_type *ft) {
    assert(num_field_types >= 0);
    if (ft == NULL) {
        return;
    }
    for (int i = 0; i < num_field_types; i++) {
        Py_XDECREF(ft[i].descr);
        ft[i].descr = NULL;
    }
    PyMem_Free(ft);
}


/*
 * Fetch custom converters for the builtin NumPy DTypes (or the generic one).
 * Structured DTypes get unpacked and `object` uses the generic method.
 *
 * TODO: This should probably be moved on the DType object in some form,
 *       to allow user DTypes to define their own converters.
 */
static set_from_ucs4_function *
get_from_ucs4_function(PyArray_Descr *descr)
{
    if (descr->type_num == NPY_BOOL) {
        return &npy_to_bool;
    }
    else if (PyDataType_ISSIGNED(descr)) {
        switch (descr->elsize) {
            case 1:
                return &npy_to_int8;
            case 2:
                return &npy_to_int16;
            case 4:
                return &npy_to_int32;
            case 8:
                return &npy_to_int64;
            default:
                assert(0);
        }
    }
    else if (PyDataType_ISUNSIGNED(descr)) {
        switch (descr->elsize) {
            case 1:
                return &npy_to_uint8;
            case 2:
                return &npy_to_uint16;
            case 4:
                return &npy_to_uint32;
            case 8:
                return &npy_to_uint64;
            default:
                assert(0);
        }
    }
    else if (descr->type_num == NPY_FLOAT) {
        return &npy_to_float;
    }
    else if (descr->type_num == NPY_DOUBLE) {
        return &npy_to_double;
    }
    else if (descr->type_num == NPY_CFLOAT) {
        return &npy_to_cfloat;
    }
    else if (descr->type_num == NPY_CDOUBLE) {
        return &npy_to_cdouble;
    }
    else if (descr->type_num == NPY_STRING) {
        return &npy_to_string;
    }
    else if (descr->type_num == NPY_UNICODE) {
        return &npy_to_unicode;
    }
    return &npy_to_generic;
}


/*
 * Note that the function cleans up `ft` on error.  If `num_field_types < 0`
 * cleanup has already happened in the internal call.
 */
static npy_intp
field_type_grow_recursive(PyArray_Descr *descr,
        npy_intp num_field_types, field_type **ft, npy_intp *ft_size,
        npy_intp field_offset)
{
    if (PyDataType_HASSUBARRAY(descr)) {
        PyArray_Dims shape = {NULL, -1};

        if (!(PyArray_IntpConverter(descr->subarray->shape, &shape))) {
             PyErr_SetString(PyExc_ValueError, "invalid subarray shape");
             field_types_xclear(num_field_types, *ft);
             return -1;
        }
        npy_intp size = PyArray_MultiplyList(shape.ptr, shape.len);
        npy_free_cache_dim_obj(shape);
        for (npy_intp i = 0; i < size; i++) {
            num_field_types = field_type_grow_recursive(descr->subarray->base,
                    num_field_types, ft, ft_size, field_offset);
            field_offset += descr->subarray->base->elsize;
            if (num_field_types < 0) {
                return -1;
            }
        }
        return num_field_types;
    }
    else if (PyDataType_HASFIELDS(descr)) {
        npy_int num_descr_fields = PyTuple_Size(descr->names);
        if (num_descr_fields < 0) {
            field_types_xclear(num_field_types, *ft);
            return -1;
        }
        for (npy_intp i = 0; i < num_descr_fields; i++) {
            PyObject *key = PyTuple_GET_ITEM(descr->names, i);
            PyObject *tup = PyObject_GetItem(descr->fields, key);
            if (tup == NULL) {
                field_types_xclear(num_field_types, *ft);
                return -1;
            }
            PyArray_Descr *field_descr;
            PyObject *title;
            int offset;
            if (!PyArg_ParseTuple(tup, "Oi|O", &field_descr, &offset, &title)) {
                Py_DECREF(tup);
                field_types_xclear(num_field_types, *ft);
                return -1;
            }
            Py_DECREF(tup);
            num_field_types = field_type_grow_recursive(
                    field_descr, num_field_types, ft, ft_size,
                    field_offset + offset);
            if (num_field_types < 0) {
                return -1;
            }
        }
        return num_field_types;
    }

    if (*ft_size <= num_field_types) {
        npy_intp alloc_size = grow_size_and_multiply(
                ft_size, 4, sizeof(field_type));
        if (alloc_size < 0) {
            field_types_xclear(num_field_types, *ft);
            return -1;
        }
        field_type *new_ft = PyMem_Realloc(*ft, alloc_size);
        if (new_ft == NULL) {
            field_types_xclear(num_field_types, *ft);
            return -1;
        }
        *ft = new_ft;
    }

    Py_INCREF(descr);
    (*ft)[num_field_types].descr = descr;
    (*ft)[num_field_types].set_from_ucs4 = get_from_ucs4_function(descr);
    (*ft)[num_field_types].structured_offset = field_offset;

    return num_field_types + 1;
}


/*
 * Prepare the "field_types" for the given dtypes/descriptors.  Currently,
 * we copy the itemsize, but the main thing is that we check for custom
 * converters.
 */
NPY_NO_EXPORT npy_intp
field_types_create(PyArray_Descr *descr, field_type **ft)
{
    if (descr->subarray != NULL) {
        /*
         * This could probably be allowed, but NumPy absorbs the dimensions
         * so it is an awkward corner case that probably never really worked.
         */
        PyErr_SetString(PyExc_TypeError,
                "file reader does not support subarray dtypes.  You can"
                "put the dtype into a structured one using "
                "`np.dtype(('name', dtype))` to avoid this limitation.");
        return -1;
    }

    npy_intp ft_size = 4;
    *ft = PyMem_Malloc(ft_size * sizeof(field_type));
    if (*ft == NULL) {
        return -1;
    }
    return field_type_grow_recursive(descr, 0, ft, &ft_size, 0);
}
