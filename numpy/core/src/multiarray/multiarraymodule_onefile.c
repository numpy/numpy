/*
 * This file includes all the .c files needed for a complete multiarray module.
 * This is used in the case where separate compilation is not enabled
 *
 * Note that the order of the includes matters
 */

#include "common.c"

#include "scalartypes.c"
#include "scalarapi.c"

#include "datetime.c"
#include "datetime_strings.c"
#include "datetime_busday.c"
#include "datetime_busdaycal.c"
#include "arraytypes.c"
#include "vdot.c"

#include "hashdescr.c"
#include "numpyos.c"

#include "descriptor.c"
#include "flagsobject.c"
#include "alloc.c"
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

#include "nditer_constr.c"
#include "nditer_api.c"
#include "nditer_templ.c"
#include "nditer_pywrap.c"
#include "lowlevel_strided_loops.c"
#include "dtype_transfer.c"
#include "einsum.c"
#include "array_assign.c"
#include "array_assign_scalar.c"
#include "array_assign_array.c"
#include "ucsnarrow.c"
#include "arrayobject.c"
#include "numpymemoryview.c"
#include "mem_overlap.c"
#include "multiarraymodule.c"
#include "compiled_base.c"

#if defined(HAVE_CBLAS)
#include "python_xerbla.c"
#include "cblasfuncs.c"
#endif
