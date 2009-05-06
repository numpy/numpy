/*
 * This file includes all the .c files needed for a complete multiarray module.
 * This is used in the case where separate compilation is not enabled
 *
 * Note that the order of the includs matters
 */

#include "common.c"

#include "arraytypes.c"

#include "hashdescr.c"
#include "numpyos.c"
#include "scalarapi.c"
#include "descriptor.c"
#include "flagsobject.c"
#include "ctors.c"
#include "iterators.c"
#include "mapping.c"
#include "number.c"
#include "getset.c"
#include "sequence.c"
#include "methods.c"
#include "convert_datatype.c"
#include "convert.c"
#include "shape.c"
#include "item_selection.c"
#include "calculation.c"
#include "usertypes.c"
#include "refcount.c"
#include "conversion_utils.c"
#include "buffer.c"

#include "scalartypes.c"

#ifndef Py_UNICODE_WIDE
#include "ucsnarrow.c"
#endif

#include "arrayobject.c"

#include "multiarraymodule.c"
