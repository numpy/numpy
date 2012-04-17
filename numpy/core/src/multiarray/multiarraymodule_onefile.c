/*
 * This file includes all the .c files needed for a complete multiarray module.
 * This is used in the case where separate compilation is not enabled
 *
 * Note that the order of the includs matters
 */

#include "common.c"

#include "scalartypes.c"
#include "scalarapi.c"

#include "datetime.c"
#include "arraytypes.c"

#include "hashdescr.c"
#include "numpyos.c"

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

#include "nditer.c"
#include "nditer_pywrap.c"
#include "lowlevel_strided_loops.c"
#include "dtype_transfer.c"
#include "einsum.c"
#include "ucsnarrow.c"

#include "arrayobject.c"

#include "numpymemoryview.c"

#include "multiarraymodule.c"
