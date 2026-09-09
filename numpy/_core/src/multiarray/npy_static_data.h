#ifndef NUMPY_CORE_SRC_MULTIARRAY_STATIC_DATA_H_
#define NUMPY_CORE_SRC_MULTIARRAY_STATIC_DATA_H_

#include <Python.h>

#include "module_state_fields.h"

#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT int
initialize_static_globals(void);

NPY_NO_EXPORT int
intern_strings(void);

NPY_NO_EXPORT int
verify_static_structs_initialized(void);

typedef struct npy_interned_str_struct {
    NPY_DECLARE_PYOBJECT_FIELDS(NPY_INTERNED_STR_FIELDS)
    PyObject *errmode_strings[NPY_ERRMODE_STRING_COUNT];
} npy_interned_str_struct;

/*
 * A struct that stores static global data used throughout
 * _multiarray_umath, mostly to cache results that would be
 * prohibitively expensive to compute at runtime in a tight loop.
 *
 * All items in this struct should be initialized during module
 * initialization and thereafter should be immutable. Mutating items in
 * this struct after module initialization is likely not thread-safe.
 */

typedef struct npy_static_pydata_struct {
    NPY_DECLARE_PYOBJECT_FIELDS(NPY_STATIC_PYDATA_FIELDS)
} npy_static_pydata_struct;


typedef struct npy_static_cdata_struct {
    /*
     * stores sys.flags.optimize as a long, which is used in the add_docstring
     * implementation
     */
    long optimize;

    /*
     * LUT used by unpack_bits
     */
    union {
        npy_uint8  bytes[8];
        npy_uint64 uint64;
    } unpack_lookup_big[256];

    /*
     * A look-up table to recover integer type numbers from type characters.
     *
     * See the _MAX_LETTER and LETTER_TO_NUM macros in arraytypes.c.src.
     *
     * The smallest type number is ?, the largest is bounded by 'z'.
     *
     * This is initialized alongside the built-in dtypes
     */
    npy_int16 _letter_to_num['z' + 1 - '?'];
} npy_static_cdata_struct;

#ifdef __cplusplus
}
#endif

#endif  // NUMPY_CORE_SRC_MULTIARRAY_STATIC_DATA_H_
