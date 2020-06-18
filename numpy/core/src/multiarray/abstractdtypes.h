#ifndef _NPY_ABSTRACTDTYPES_H
#define _NPY_ABSTRACTDTYPES_H

#include "dtypemeta.h"

/*
 * These are mainly needed for value based promotion in ufuncs.  It
 * may be necessary to make them (partially) public, to allow user-defined
 * dtypes to perform value based casting.
 */
NPY_NO_EXPORT PyTypeObject PyArrayAbstractObjDTypeMeta_Type;
NPY_NO_EXPORT PyArray_DTypeMeta PyArray_PyIntAbstractDType;
NPY_NO_EXPORT PyArray_DTypeMeta PyArray_PyFloatAbstractDType;
NPY_NO_EXPORT PyArray_DTypeMeta PyArray_PyComplexAbstractDType;

NPY_NO_EXPORT int
initialize_abstract_dtypes_and_map_others();

#endif  /*_NPY_ABSTRACTDTYPES_H */
