#if NPY_BITSOF_LONG == 8
#  ifndef PyInt8ScalarObject
#    define PyInt8ScalarObject PyLongScalarObject
#    define PyInt8ArrType_Type PyLongArrType_Type
#  endif
#elif NPY_BITSOF_LONG == 16
#  ifndef PyInt16ScalarObject
#    define PyInt16ScalarObject PyLongScalarObject
#    define PyInt16ArrType_Type PyLongArrType_Type
#  endif
#elif NPY_BITSOF_LONG == 32
#  ifndef PyInt32ScalarObject
#    define PyInt32ScalarObject PyLongScalarObject
#    define PyInt32ArrType_Type PyLongArrType_Type
#  endif
#elif NPY_BITSOF_LONG == 64
#  ifndef PyInt64ScalarObject
#    define PyInt64ScalarObject PyLongScalarObject
#    define PyInt64ArrType_Type PyLongArrType_Type
#  endif
#elif NPY_BITSOF_LONG == 128
#  ifndef PyInt128ScalarObject
#    define PyInt128ScalarObject PyLongScalarObject
#    define PyInt128ArrType_Type PyLongArrType_Type
#  endif
#endif

#if NPY_BITSOF_LONGLONG == 8
#  ifndef PyInt8ScalarObject
#    define PyInt8ScalarObject PyLongLongScalarObject
#    define PyInt8ArrType_Type PyLongLongArrType_Type
#  endif
#elif NPY_BITSOF_LONGLONG == 16
#  ifndef PyInt16ScalarObject
#    define PyInt16ScalarObject PyLongLongScalarObject
#    define PyInt16ArrType_Type PyLongLongArrType_Type
#  endif
#elif NPY_BITSOF_LONGLONG == 32
#  ifndef PyInt32ScalarObject
#    define PyInt32ScalarObject PyLongLongScalarObject
#    define PyInt32ArrType_Type PyLongLongArrType_Type
#  endif
#elif NPY_BITSOF_LONGLONG == 64
#  ifndef PyInt64ScalarObject
#    define PyInt64ScalarObject PyLongLongScalarObject
#    define PyInt64ArrType_Type PyLongLongArrType_Type
#  endif
#elif NPY_BITSOF_LONGLONG == 128
#  ifndef PyInt128ScalarObject
#    define PyInt128ScalarObject PyLongLongScalarObject
#    define PyInt128ArrType_Type PyLongLongArrType_Type
#  endif
#elif NPY_BITSOF_LONGLONG == 256
#  ifndef PyInt256ScalarObject
#    define PyInt256ScalarObject PyLongLongScalarObject
#    define PyInt256ArrType_Type PyLongLongArrType_Type
#  endif
#endif

#if NPY_BITSOF_INT == 8
#  ifndef PyInt8ScalarObject
#    define PyInt8ScalarObject PyIntScalarObject
#    define PyInt8ArrType_Type PyIntArrType_Type
#  endif
#elif NPY_BITSOF_INT == 16
#  ifndef PyInt16ScalarObject
#    define PyInt16ScalarObject PyIntScalarObject
#    define PyInt16ArrType_Type PyIntArrType_Type
#  endif
#elif NPY_BITSOF_INT == 32
#  ifndef PyInt32ScalarObject
#    define PyInt32ScalarObject PyIntScalarObject
#    define PyInt32ArrType_Type PyIntArrType_Type
#  endif
#elif NPY_BITSOF_INT == 64
#  ifndef PyInt64ScalarObject
#    define PyInt64ScalarObject PyIntScalarObject
#    define PyInt64ArrType_Type PyIntArrType_Type
#  endif
#elif NPY_BITSOF_INT == 128
#  ifndef PyInt128ScalarObject
#    define PyInt128ScalarObject PyIntScalarObject
#    define PyInt128ArrType_Type PyIntArrType_Type
#  endif
#endif

#if NPY_BITSOF_SHORT == 8
#  ifndef PyInt8ScalarObject
#    define PyInt8ScalarObject PyShortScalarObject
#    define PyInt8ArrType_Type PyShortArrType_Type
#  endif
#elif NPY_BITSOF_SHORT == 16
#  ifndef PyInt16ScalarObject
#    define PyInt16ScalarObject PyShortScalarObject
#    define PyInt16ArrType_Type PyShortArrType_Type
#  endif
#elif NPY_BITSOF_SHORT == 32
#  ifndef PyInt32ScalarObject
#    define PyInt32ScalarObject PyShortScalarObject
#    define PyInt32ArrType_Type PyShortArrType_Type
#  endif
#elif NPY_BITSOF_SHORT == 64
#  ifndef PyInt64ScalarObject
#    define PyInt64ScalarObject PyShortScalarObject
#    define PyInt64ArrType_Type PyShortArrType_Type
#  endif
#elif NPY_BITSOF_SHORT == 128
#  ifndef PyInt128ScalarObject
#    define PyInt128ScalarObject PyShortScalarObject
#    define PyInt128ArrType_Type PyShortArrType_Type
#  endif
#endif

#if NPY_BITSOF_CHAR == 8
#  ifndef PyInt8ScalarObject
#    define PyInt8ScalarObject PyByteScalarObject
#    define PyInt8ArrType_Type PyByteArrType_Type
#  endif
#elif NPY_BITSOF_CHAR == 16
#  ifndef PyInt16ScalarObject
#    define PyInt16ScalarObject PyByteScalarObject
#    define PyInt16ArrType_Type PyByteArrType_Type
#  endif
#elif NPY_BITSOF_CHAR == 32
#  ifndef PyInt32ScalarObject
#    define PyInt32ScalarObject PyByteScalarObject
#    define PyInt32ArrType_Type PyByteArrType_Type
#  endif
#elif NPY_BITSOF_CHAR == 64
#  ifndef PyInt64ScalarObject
#    define PyInt64ScalarObject PyByteScalarObject
#    define PyInt64ArrType_Type PyByteArrType_Type
#  endif
#elif NPY_BITSOF_CHAR == 128
#  ifndef PyInt128ScalarObject
#    define PyInt128ScalarObject PyByteScalarObject
#    define PyInt128ArrType_Type PyByteArrType_Type
#  endif
#endif

#if NPY_BITSOF_DOUBLE == 16
#  ifndef PyFloat16ScalarObject
#    define PyFloat16ScalarObject PyDoubleScalarObject
#    define PyComplex32ScalarObject PyCDoubleScalarObject
#    define PyFloat16ArrType_Type PyDoubleArrType_Type
#    define PyComplex32ArrType_Type PyCDoubleArrType_Type
#  endif
#elif NPY_BITSOF_DOUBLE == 32
#  ifndef PyFloat32ScalarObject
#    define PyFloat32ScalarObject PyDoubleScalarObject
#    define PyComplex64ScalarObject PyCDoubleScalarObject
#    define PyFloat32ArrType_Type PyDoubleArrType_Type
#    define PyComplex64ArrType_Type PyCDoubleArrType_Type
#  endif
#elif NPY_BITSOF_DOUBLE == 64
#  ifndef PyFloat64ScalarObject
#    define PyFloat64ScalarObject PyDoubleScalarObject
#    define PyComplex128ScalarObject PyCDoubleScalarObject
#    define PyFloat64ArrType_Type PyDoubleArrType_Type
#    define PyComplex128ArrType_Type PyCDoubleArrType_Type
#  endif
#elif NPY_BITSOF_DOUBLE == 80
#  ifndef PyFloat80ScalarObject
#    define PyFloat80ScalarObject PyDoubleScalarObject
#    define PyComplex160ScalarObject PyCDoubleScalarObject
#    define PyFloat80ArrType_Type PyDoubleArrType_Type
#    define PyComplex160ArrType_Type PyCDoubleArrType_Type
#  endif
#elif NPY_BITSOF_DOUBLE == 96
#  ifndef PyFloat96ScalarObject
#    define PyFloat96ScalarObject PyDoubleScalarObject
#    define PyComplex192ScalarObject PyCDoubleScalarObject
#    define PyFloat96ArrType_Type PyDoubleArrType_Type
#    define PyComplex192ArrType_Type PyCDoubleArrType_Type
#  endif
#elif NPY_BITSOF_DOUBLE == 128
#  ifndef PyFloat128ScalarObject
#    define PyFloat128ScalarObject PyDoubleScalarObject
#    define PyComplex256ScalarObject PyCDoubleScalarObject
#    define PyFloat128ArrType_Type PyDoubleArrType_Type
#    define PyComplex256ArrType_Type PyCDoubleArrType_Type
#  endif
#endif

#if NPY_BITSOF_FLOAT == 16
#  ifndef PyFloat16ScalarObject
#    define PyFloat16ScalarObject PyFloatScalarObject
#    define PyComplex32ScalarObject PyCFloatScalarObject
#    define PyFloat16ArrType_Type PyFloatArrType_Type
#    define PyComplex32ArrType_Type PyCFloatArrType_Type
#  endif
#elif NPY_BITSOF_FLOAT == 32
#  ifndef PyFloat32ScalarObject
#    define PyFloat32ScalarObject PyFloatScalarObject
#    define PyComplex64ScalarObject PyCFloatScalarObject
#    define PyFloat32ArrType_Type PyFloatArrType_Type
#    define PyComplex64ArrType_Type PyCFloatArrType_Type
#  endif
#elif NPY_BITSOF_FLOAT == 64
#  ifndef PyFloat64ScalarObject
#    define PyFloat64ScalarObject PyFloatScalarObject
#    define PyComplex128ScalarObject PyCFloatScalarObject
#    define PyFloat64ArrType_Type PyFloatArrType_Type
#    define PyComplex128ArrType_Type PyCFloatArrType_Type
#  endif
#elif NPY_BITSOF_FLOAT == 80
#  ifndef PyFloat80ScalarObject
#    define PyFloat80ScalarObject PyFloatScalarObject
#    define PyComplex160ScalarObject PyCFloatScalarObject
#    define PyFloat80ArrType_Type PyFloatArrType_Type
#    define PyComplex160ArrType_Type PyCFloatArrType_Type
#  endif
#elif NPY_BITSOF_FLOAT == 96
#  ifndef PyFloat96ScalarObject
#    define PyFloat96ScalarObject PyFloatScalarObject
#    define PyComplex192ScalarObject PyCFloatScalarObject
#    define PyFloat96ArrType_Type PyFloatArrType_Type
#    define PyComplex192ArrType_Type PyCFloatArrType_Type
#  endif
#elif NPY_BITSOF_FLOAT == 128
#  ifndef PyFloat128ScalarObject
#    define PyFloat128ScalarObject PyFloatScalarObject
#    define PyComplex256ScalarObject PyCFloatScalarObject
#    define PyFloat128ArrType_Type PyFloatArrType_Type
#    define PyComplex256ArrType_Type PyCFloatArrType_Type
#  endif
#endif

#if NPY_BITSOF_LONGDOUBLE == 16
#  ifndef PyFloat16ScalarObject
#    define PyFloat16ScalarObject PyLongDoubleScalarObject
#    define PyComplex32ScalarObject PyCLongDoubleScalarObject
#    define PyFloat16ArrType_Type PyLongDoubleArrType_Type
#    define PyComplex32ArrType_Type PyCLongDoubleArrType_Type
#  endif
#elif NPY_BITSOF_LONGDOUBLE == 32
#  ifndef PyFloat32ScalarObject
#    define PyFloat32ScalarObject PyLongDoubleScalarObject
#    define PyComplex64ScalarObject PyCLongDoubleScalarObject
#    define PyFloat32ArrType_Type PyLongDoubleArrType_Type
#    define PyComplex64ArrType_Type PyCLongDoubleArrType_Type
#  endif
#elif NPY_BITSOF_LONGDOUBLE == 64
#  ifndef PyFloat64ScalarObject
#    define PyFloat64ScalarObject PyLongDoubleScalarObject
#    define PyComplex128ScalarObject PyCLongDoubleScalarObject
#    define PyFloat64ArrType_Type PyLongDoubleArrType_Type
#    define PyComplex128ArrType_Type PyCLongDoubleArrType_Type
#  endif
#elif NPY_BITSOF_LONGDOUBLE == 80
#  ifndef PyFloat80ScalarObject
#    define PyFloat80ScalarObject PyLongDoubleScalarObject
#    define PyComplex160ScalarObject PyCLongDoubleScalarObject
#    define PyFloat80ArrType_Type PyLongDoubleArrType_Type
#    define PyComplex160ArrType_Type PyCLongDoubleArrType_Type
#  endif
#elif NPY_BITSOF_LONGDOUBLE == 96
#  ifndef PyFloat96ScalarObject
#    define PyFloat96ScalarObject PyLongDoubleScalarObject
#    define PyComplex192ScalarObject PyCLongDoubleScalarObject
#    define PyFloat96ArrType_Type PyLongDoubleArrType_Type
#    define PyComplex192ArrType_Type PyCLongDoubleArrType_Type
#  endif
#elif NPY_BITSOF_LONGDOUBLE == 128
#  ifndef PyFloat128ScalarObject
#    define PyFloat128ScalarObject PyLongDoubleScalarObject
#    define PyComplex256ScalarObject PyCLongDoubleScalarObject
#    define PyFloat128ArrType_Type PyLongDoubleArrType_Type
#    define PyComplex256ArrType_Type PyCLongDoubleArrType_Type
#  endif
#elif NPY_BITSOF_LONGDOUBLE == 256
#  ifndef PyFloat256ScalarObject
#    define PyFloat256ScalarObject PyLongDoubleScalarObject
#    define PyComplex512ScalarObject PyCLongDoubleScalarObject
#    define PyFloat256ArrType_Type PyLongDoubleArrType_Type
#    define PyComplex512ArrType_Type PyCLongDoubleArrType_Type
#  endif
#endif

