/*
 * module_state.h — Per-module state for _multiarray_umath
 *
 * This header defines the master state struct that Python allocates once per
 * interpreter when _multiarray_umath is imported. All global PyObject* caches for the _multiarray_umath module must eventually live here instead of as process-global variables.
 *
 * Migration status (FIXME: update this as each struct is moved):
 *   [x] npy_interned_str    — migrated to multiarray_umath_state.interned_str
 *   [x] npy_static_pydata   — migrated to multiarray_umath_state.static_pydata
 *   [x] npy_static_cdata    — migrated to multiarray_umath_state.static_cdata
 *   [x] npy_runtime_imports — migrated to multiarray_umath_state.runtime_imports
 *                             (import mutex remains process-global in npy_import.c)
 *   [x] npy_global_state    — migrated to multiarray_umath_state.global_state
 *   [x] typeDict (descriptor.c)                    — migrated to multiarray_umath_state.typeDict
 *   [x] current_handler (alloc.c)                  — migrated to multiarray_umath_state.current_handler
 *   [x] _global_pytype_to_type_dict (array_coercion.c) — migrated to multiarray_umath_state.global_pytype_to_type_dict
 *   [x] n_ops (number.c)                           — migrated to multiarray_umath_state.n_ops
 *   [x] uo_index (alloc.c) — dead code, deleted instead of migrated
 */

#ifndef NUMPY_CORE_SRC_MULTIARRAY_MODULE_STATE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MODULE_STATE_H_

#include "npy_static_data.h"   /* npy_interned_str_struct, npy_static_pydata_struct,
                                   npy_static_cdata_struct */
#include "npy_import.h"        /* npy_runtime_imports_struct */
#include "multiarraymodule.h"  /* npy_global_state_struct */
#include "number.h"             /* NumericOps */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Master module state struct.
 *
 * Python allocates sizeof(multiarray_umath_state) bytes per interpreter
 * instance and zero-initializes it. Access via get_module_state() below.
 *
 * NOTE: The sub-struct fields here are placeholders for when each global
 * is actually migrated. Until then the real data lives in the process-global
 * variables and the fields in this struct are unused.
 */
typedef struct {
    npy_interned_str_struct    interned_str;
    npy_static_pydata_struct   static_pydata;
    npy_static_cdata_struct    static_cdata;
    npy_runtime_imports_struct runtime_imports;
    npy_global_state_struct    global_state;

    /* scattered globals — formerly descriptor.c/alloc.c/array_coercion.c/number.c globals */
    PyObject *typeDict;
    PyObject *current_handler;
    PyObject *global_pytype_to_type_dict;
    NumericOps n_ops;
} multiarray_umath_state;

/*
 * TRANSITIONAL: process-global pointer to the module state.
 *
 * Set once during module init in _multiarray_umath_exec(). Used by deep
 * internal functions that don't have easy access to the module pointer.
 *
 * FIXME: Remove this once all access sites are updated to receive the
 * module/state pointer via proper channels (threading or type methods).
 */
NPY_VISIBILITY_HIDDEN extern multiarray_umath_state *_npy_module_state;

/*
 * Get module state from the module object.
 */
static inline multiarray_umath_state *
get_module_state(PyObject *module)
{
    void *state = PyModule_GetState(module);
    assert(state != NULL);
    return (multiarray_umath_state *)state;
}

/*
 * TRANSITIONAL: Get module state without a module pointer.
 *
 * Use this only in internal functions deep in the call chain that cannot
 * easily receive a module pointer yet. Prefer get_module_state() where
 * a module pointer is available.
 *
 * FIXME: Remove all call sites once the full migration is complete.
 */
static inline multiarray_umath_state *
npy_get_module_state(void)
{
    assert(_npy_module_state != NULL);
    return _npy_module_state;
}

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MODULE_STATE_H_ */
