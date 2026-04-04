/*
 * Public exposure of the DType Classes.  These are tricky to expose
 * via the Python API, so they are exposed through this header for now.
 *
 * These definitions are only relevant for the public API and we reserve
 * the slots 320-360 in the API table generation for this (currently).
 *
 * TODO: This file should be consolidated with the API table generation
 *       (although not sure the current generation is worth preserving).
 */
#ifndef NUMPY_CORE_INCLUDE_NUMPY__PUBLIC_DTYPE_API_TABLE_H_
#define NUMPY_CORE_INCLUDE_NUMPY__PUBLIC_DTYPE_API_TABLE_H_

#if !(defined(NPY_INTERNAL_BUILD) && NPY_INTERNAL_BUILD)

/* All of these require NumPy 2.0 support */
#if NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION

/*
 * The type of the DType metaclass. At some point deprecate these
 * non-pointer definitions
 */
#define PyArrayDTypeMeta_Type (*(PyTypeObject *)(PyArray_API + 320)[0])
/*
 * NumPy's builtin DTypes:
 */
#define PyArray_BoolDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[1])
/* Integers */
#define PyArray_ByteDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[2])
#define PyArray_UByteDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[3])
#define PyArray_ShortDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[4])
#define PyArray_UShortDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[5])
#define PyArray_IntDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[6])
#define PyArray_UIntDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[7])
#define PyArray_LongDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[8])
#define PyArray_ULongDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[9])
#define PyArray_LongLongDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[10])
#define PyArray_ULongLongDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[11])
/* Integer aliases */
#define PyArray_Int8DType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[12])
#define PyArray_UInt8DType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[13])
#define PyArray_Int16DType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[14])
#define PyArray_UInt16DType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[15])
#define PyArray_Int32DType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[16])
#define PyArray_UInt32DType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[17])
#define PyArray_Int64DType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[18])
#define PyArray_UInt64DType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[19])
#define PyArray_IntpDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[20])
#define PyArray_UIntpDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[21])
/* Floats */
#define PyArray_HalfDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[22])
#define PyArray_FloatDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[23])
#define PyArray_DoubleDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[24])
#define PyArray_LongDoubleDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[25])
/* Complex */
#define PyArray_CFloatDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[26])
#define PyArray_CDoubleDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[27])
#define PyArray_CLongDoubleDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[28])
/* String/Bytes */
#define PyArray_BytesDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[29])
#define PyArray_UnicodeDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[30])
/* Datetime/Timedelta */
#define PyArray_DatetimeDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[31])
#define PyArray_TimedeltaDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[32])
/* Object/Void */
#define PyArray_ObjectDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[33])
#define PyArray_VoidDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[34])
/* Python types (used as markers for scalars) */
#define PyArray_PyLongDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[35])
#define PyArray_PyFloatDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[36])
#define PyArray_PyComplexDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[37])
/* Default integer type */
#define PyArray_DefaultIntDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[38])
/* New non-legacy DTypes follow in the order they were added */
#define PyArray_StringDType (*(PyArray_DTypeMeta *)(PyArray_API + 320)[39])

/* NOTE: offset 40 is free */

/* Need to start with a larger offset again for the abstract classes: */
#define PyArray_IntAbstractDType (*(PyArray_DTypeMeta *)PyArray_API[366])
#define PyArray_FloatAbstractDType (*(PyArray_DTypeMeta *)PyArray_API[367])
#define PyArray_ComplexAbstractDType (*(PyArray_DTypeMeta *)PyArray_API[368])

#endif /* NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION */

#if NPY_FEATURE_VERSION >= NPY_2_5_API_VERSION

/*
 * Pointer to the type of the DType metaclass
 */
#define PyArrayDTypeMeta_TypePtr (PyTypeObject *)(PyArray_API + 320)[0]
/*
 * NumPy's builtin DTypes:
 */
#define PyArray_BoolDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[1]
/* Integers */
#define PyArray_ByteDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[2]
#define PyArray_UByteDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[3]
#define PyArray_ShortDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[4]
#define PyArray_UShortDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[5]
#define PyArray_IntDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[6]
#define PyArray_UIntDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[7]
#define PyArray_LongDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[8]
#define PyArray_ULongDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[9]
#define PyArray_LongLongDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[10]
#define PyArray_ULongLongDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[11]
/* Integer aliases */
#define PyArray_Int8DTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[12]
#define PyArray_UInt8DTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[13]
#define PyArray_Int16DTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[14]
#define PyArray_UInt16DTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[15]
#define PyArray_Int32DTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[16]
#define PyArray_UInt32DTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[17]
#define PyArray_Int64DTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[18]
#define PyArray_UInt64DTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[19]
#define PyArray_IntpDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[20]
#define PyArray_UIntpDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[21]
/* Floats */
#define PyArray_HalfDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[22]
#define PyArray_FloatDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[23]
#define PyArray_DoubleDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[24]
#define PyArray_LongDoubleDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[25]
/* Complex */
#define PyArray_CFloatDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[26]
#define PyArray_CDoubleDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[27]
#define PyArray_CLongDoubleDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[28]
/* String/Bytes */
#define PyArray_BytesDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[29]
#define PyArray_UnicodeDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[30]
/* Datetime/Timedelta */
#define PyArray_DatetimeDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[31]
#define PyArray_TimedeltaDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[32]
/* Object/Void */
#define PyArray_ObjectDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[33]
#define PyArray_VoidDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[34]
/* Python types (used as markers for scalars) */
#define PyArray_PyLongDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[35]
#define PyArray_PyFloatDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[36]
#define PyArray_PyComplexDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[37]
/* Default integer type */
#define PyArray_DefaultIntDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[38]
/* New non-legacy DTypes follow in the order they were added */
#define PyArray_StringDTypePtr (PyArray_DTypeMeta *)(PyArray_API + 320)[39]

/* NOTE: offset 40 is free */

/* Need to start with a larger offset again for the abstract classes: */
#define PyArray_IntAbstractDTypePtr (PyArray_DTypeMeta *)PyArray_API[366]
#define PyArray_FloatAbstractDTypePtr (PyArray_DTypeMeta *)PyArray_API[367]
#define PyArray_ComplexAbstractDTypePtr (PyArray_DTypeMeta *)PyArray_API[368]

#endif /* NPY_FEATURE_VERSION >= NPY_2_5_API_VERSION */

#endif  /* NPY_INTERNAL_BUILD */
#endif  /* NUMPY_CORE_INCLUDE_NUMPY__PUBLIC_DTYPE_API_TABLE_H_ */
