#ifndef _NPY_ABSTRACTDTYPES_H
#define _NPY_ABSTRACTDTYPES_H

#include "dtypemeta.h"

/*
 * These are mainly needed for value based promotion in ufuncs.  It
 * may be necessary to make them (partially) public, to allow user-defined
 * dtypes to perform value based casting.
 */
NPY_NO_EXPORT extern PyTypeObject PyArrayAbstractObjDTypeMeta_Type;
NPY_NO_EXPORT extern PyArray_DTypeMeta PyArray_PyIntAbstractDType;
NPY_NO_EXPORT extern PyArray_DTypeMeta PyArray_PyFloatAbstractDType;
NPY_NO_EXPORT extern PyArray_DTypeMeta PyArray_PyComplexAbstractDType;

NPY_NO_EXPORT int
initialize_and_map_pytypes_to_dtypes(void);

#endif  /*_NPY_ABSTRACTDTYPES_H */
