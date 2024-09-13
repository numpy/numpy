
#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_FIELD_TYPES_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_FIELD_TYPES_H_

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdint.h>
#include <stdbool.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/ndarraytypes.h"

#include "textreading/parser_config.h"

/**
 * Function defining the conversion for each value.
 *
 * This function must support unaligned memory access.  As of now, there is
 * no special error handling (in whatever form):  We assume that it is always
 * reasonable to raise a `ValueError` noting the string that failed to be
 * converted.
 *
 * NOTE: An earlier version of the code had unused default values (pandas
 *       does this) when columns are missing.  We could define this either
 *       by passing `NULL` in, or by adding a default explicitly somewhere.
 *       (I think users should probably have to define the default, at which
 *       point it doesn't matter here.)
 *
 * NOTE: We are currently passing the parser config, this could be made public
 *       or could be set up to be dtype specific/private.  Always passing
 *       pconfig fully seems easier right now even if it may change.
 *       (A future use-case may for example be user-specified strings that are
 *       considered boolean True or False).
 *
 * TODO: Aside from nailing down the above notes, it may be nice to expose
 *       these function publicly.  This could allow user DTypes to provide
 *       a converter or custom converters written in C rather than Python.
 *
 * @param descr The NumPy descriptor of the field (may be byte-swapped, etc.)
 * @param str Pointer to the beginning of the UCS4 string to be parsed.
 * @param end Pointer to the end of the UCS4 string.  This value is currently
 *            guaranteed to be `\0`, ensuring that parsers can rely on
 *            nul-termination.
 * @param dataptr The pointer where to store the parsed value
 * @param pconfig Additional configuration for the parser.
 * @returns 0 on success and -1 on failure.  If the return value is -1 an
 *          error may or may not be set.  If an error is set, it is chained
 *          behind the generic ValueError.
 */
typedef int (set_from_ucs4_function)(
        PyArray_Descr *descr, const Py_UCS4 *str, const Py_UCS4 *end,
        char *dataptr, parser_config *pconfig);

typedef struct _field_type {
    set_from_ucs4_function *set_from_ucs4;
    /* The original NumPy descriptor */
    PyArray_Descr *descr;
    /* Offset to this entry within row. */
    npy_intp structured_offset;
} field_type;


NPY_NO_EXPORT void
field_types_xclear(int num_field_types, field_type *ft);

NPY_NO_EXPORT npy_intp
field_types_create(PyArray_Descr *descr, field_type **ft);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_FIELD_TYPES_H_ */
