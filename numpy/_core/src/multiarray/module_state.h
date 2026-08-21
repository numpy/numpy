#ifndef NUMPY_CORE_SRC_MULTIARRAY_MODULE_STATE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MODULE_STATE_H_

#include <stddef.h>  /* for offsetof */

#include "npy_static_data.h"
#include "npy_import.h"
#include "multiarraymodule.h"
#include "number.h"
#include "module_state_fields.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    npy_interned_str_struct    interned_str;
    npy_static_pydata_struct   static_pydata;
    npy_static_cdata_struct    static_cdata;
    npy_runtime_imports_struct runtime_imports;
    npy_global_state_struct    global_state;

    PyObject *typeDict;
    PyObject *current_handler;
    PyObject *global_pytype_to_type_dict;
    NumericOps n_ops;
} multiarray_umath_state;

/* All members are PyObject *, so these catch one missing from a field list. */
static_assert(sizeof(npy_interned_str_struct) ==
        (NPY_FIELD_COUNT(NPY_INTERNED_STR_FIELDS) + NPY_ERRMODE_STRING_COUNT)
                * sizeof(PyObject *),
        "npy_interned_str_struct member missing from NPY_INTERNED_STR_FIELDS");

static_assert(sizeof(npy_static_pydata_struct) ==
        NPY_FIELD_COUNT(NPY_STATIC_PYDATA_FIELDS) * sizeof(PyObject *),
        "npy_static_pydata_struct member missing from "
        "NPY_STATIC_PYDATA_FIELDS");

static_assert(sizeof(npy_runtime_imports_struct) ==
        NPY_FIELD_COUNT(NPY_RUNTIME_IMPORTS_FIELDS) * sizeof(PyObject *),
        "npy_runtime_imports_struct member missing from "
        "NPY_RUNTIME_IMPORTS_FIELDS");

static_assert(sizeof(NumericOps) ==
        NPY_FIELD_COUNT(NPY_N_OPS_FIELDS) * sizeof(PyObject *),
        "NumericOps member missing from NPY_N_OPS_FIELDS");

/* The loose members sit contiguously between the sub-structs and n_ops. */
static_assert(offsetof(multiarray_umath_state, n_ops) -
        offsetof(multiarray_umath_state, typeDict) ==
        NPY_FIELD_COUNT(NPY_MODULE_STATE_OBJECT_FIELDS) * sizeof(PyObject *),
        "multiarray_umath_state member missing from "
        "NPY_MODULE_STATE_OBJECT_FIELDS");

static inline multiarray_umath_state *
get_module_state(PyObject *module)
{
    void *state = PyModule_GetState(module);
    assert(state != NULL);
    return (multiarray_umath_state *)state;
}

/*
 * Only one state ever exists: the module opts out of subinterpreters and
 * refuses a second load. Use only where no module pointer is reachable.
 *
 * FIXME: Remove once all access sites receive the module or state pointer.
 */
NPY_VISIBILITY_HIDDEN extern multiarray_umath_state *_npy_module_state;

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MODULE_STATE_H_ */
