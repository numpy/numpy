
#ifndef _FIELD_TYPES_H_
#define _FIELD_TYPES_H_

#include <stdint.h>
#include <stdbool.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/ndarraytypes.h"

#include "textreading/parser_config.h"

/*
 * The original code had some error details, but I assume that we don't need
 * it.  Printing the string from which we tried to modify it should be fine.
 * This should potentially be public NumPy API, although it is tricky, NumPy
 *
 * This function must support unaligned memory access.
 *
 * NOTE: An earlier version of the code had unused default versions (pandas
 *       does this) when columns are missing.  We could define this either
 *       by passing `NULL` in, or by adding a default explicitly somewhere.
 *       (I think users should probably have to define the default, at which
 *       point it doesn't matter here.)
 *
 * NOTE: We are currently passing the parser config, this could be made public
 *       or could be set up to be dtype specific/private.  Always passing
 *       pconfig fully seems easier right now even if it may change.
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


void
field_types_xclear(int num_field_types, field_type *ft);

npy_intp
field_types_create(PyArray_Descr *descr, field_type **ft);

#endif
