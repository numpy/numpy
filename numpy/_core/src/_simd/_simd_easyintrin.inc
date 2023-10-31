/**
 * This file is included by `_simd.dispatch.c.src`. Its contents are affected by the simd configuration, and
 * therefore must be built multiple times. Making it a standalone `.c` file with `NPY_VISIBILITY_HIDDEN`
 * symbols would require judicious use of `NPY_CPU_DISPATCH_DECLARE` and `NPY_CPU_DISPATCH_CURFX`, which was
 * deemed too harmful to readability.
 */
#define SIMD_INTRIN_DEF(NAME) \
    { NPY_TOSTRING(NAME), simd__intrin_##NAME, METH_VARARGS, NULL } , // comma

#define SIMD_IMPL_INTRIN_0(NAME, RET)                     \
    static PyObject *simd__intrin_##NAME                  \
    (PyObject* NPY_UNUSED(self), PyObject *args)          \
    {                                                     \
        if (!PyArg_ParseTuple(                            \
            args, ":" NPY_TOSTRING(NAME))                 \
        ) return NULL;                                    \
        simd_arg a = {                                    \
            .dtype = simd_data_##RET,                     \
            .data  = {.RET = npyv_##NAME()},              \
        };                                                \
        return simd_arg_to_obj(&a);                       \
    }

#define SIMD_IMPL_INTRIN_0N(NAME)                         \
    static PyObject *simd__intrin_##NAME                  \
    (PyObject* NPY_UNUSED(self), PyObject *args)          \
    {                                                     \
        if (!PyArg_ParseTuple(                            \
            args, ":" NPY_TOSTRING(NAME))                 \
        ) return NULL;                                    \
        npyv_##NAME();                                    \
        Py_RETURN_NONE;                                   \
    }

#define SIMD_IMPL_INTRIN_1(NAME, RET, IN0)                \
    static PyObject *simd__intrin_##NAME                  \
    (PyObject* NPY_UNUSED(self), PyObject *args)          \
    {                                                     \
        simd_arg arg = {.dtype = simd_data_##IN0};        \
        if (!PyArg_ParseTuple(                            \
            args, "O&:"NPY_TOSTRING(NAME),                \
            simd_arg_converter, &arg                      \
        )) return NULL;                                   \
        simd_data data = {.RET = npyv_##NAME(             \
            arg.data.IN0                                  \
        )};                                               \
        simd_arg_free(&arg);                              \
        simd_arg ret = {                                  \
            .data = data, .dtype = simd_data_##RET        \
        };                                                \
        return simd_arg_to_obj(&ret);                     \
    }

#define SIMD_IMPL_INTRIN_2(NAME, RET, IN0, IN1)           \
    static PyObject *simd__intrin_##NAME                  \
    (PyObject* NPY_UNUSED(self), PyObject *args)          \
    {                                                     \
        simd_arg arg1 = {.dtype = simd_data_##IN0};       \
        simd_arg arg2 = {.dtype = simd_data_##IN1};       \
        if (!PyArg_ParseTuple(                            \
            args, "O&O&:"NPY_TOSTRING(NAME),              \
            simd_arg_converter, &arg1,                    \
            simd_arg_converter, &arg2                     \
        )) return NULL;                                   \
        simd_data data = {.RET = npyv_##NAME(             \
            arg1.data.IN0, arg2.data.IN1                  \
        )};                                               \
        simd_arg_free(&arg1);                             \
        simd_arg_free(&arg2);                             \
        simd_arg ret = {                                  \
            .data = data, .dtype = simd_data_##RET        \
        };                                                \
        return simd_arg_to_obj(&ret);                     \
    }

#define SIMD__REPEAT_2IMM(C, NAME, IN0) \
    C == arg2.data.u8 ? NPY_CAT(npyv_, NAME)(arg1.data.IN0, C) :

#define SIMD_IMPL_INTRIN_2IMM(NAME, RET, IN0, CONST_RNG)  \
    static PyObject *simd__intrin_##NAME                  \
    (PyObject* NPY_UNUSED(self), PyObject *args)          \
    {                                                     \
        simd_arg arg1 = {.dtype = simd_data_##IN0};       \
        simd_arg arg2 = {.dtype = simd_data_u8};          \
        if (!PyArg_ParseTuple(                            \
            args, "O&O&:"NPY_TOSTRING(NAME),              \
            simd_arg_converter, &arg1,                    \
            simd_arg_converter, &arg2                     \
        )) return NULL;                                   \
        simd_data data = {.u64 = 0};                      \
        data.RET = NPY_CAT(SIMD__IMPL_COUNT_, CONST_RNG)( \
            SIMD__REPEAT_2IMM, NAME, IN0                  \
        ) data.RET;                                       \
        simd_arg_free(&arg1);                             \
        simd_arg ret = {                                  \
            .data = data, .dtype = simd_data_##RET        \
        };                                                \
        return simd_arg_to_obj(&ret);                     \
    }

#define SIMD_IMPL_INTRIN_3(NAME, RET, IN0, IN1, IN2)      \
    static PyObject *simd__intrin_##NAME                  \
    (PyObject* NPY_UNUSED(self), PyObject *args)          \
    {                                                     \
        simd_arg arg1 = {.dtype = simd_data_##IN0};       \
        simd_arg arg2 = {.dtype = simd_data_##IN1};       \
        simd_arg arg3 = {.dtype = simd_data_##IN2};       \
        if (!PyArg_ParseTuple(                            \
            args, "O&O&O&:"NPY_TOSTRING(NAME),            \
            simd_arg_converter, &arg1,                    \
            simd_arg_converter, &arg2,                    \
            simd_arg_converter, &arg3                     \
        )) return NULL;                                   \
        simd_data data = {.RET = npyv_##NAME(             \
            arg1.data.IN0, arg2.data.IN1,                 \
            arg3.data.IN2                                 \
        )};                                               \
        simd_arg_free(&arg1);                             \
        simd_arg_free(&arg2);                             \
        simd_arg_free(&arg3);                             \
        simd_arg ret = {                                  \
            .data = data, .dtype = simd_data_##RET        \
        };                                                \
        return simd_arg_to_obj(&ret);                     \
    }

#define SIMD_IMPL_INTRIN_4(NAME, RET, IN0, IN1, IN2, IN3) \
    static PyObject *simd__intrin_##NAME                  \
    (PyObject* NPY_UNUSED(self), PyObject *args)          \
    {                                                     \
        simd_arg arg1 = {.dtype = simd_data_##IN0};       \
        simd_arg arg2 = {.dtype = simd_data_##IN1};       \
        simd_arg arg3 = {.dtype = simd_data_##IN2};       \
        simd_arg arg4 = {.dtype = simd_data_##IN3};       \
        if (!PyArg_ParseTuple(                            \
            args, "O&O&O&O&:"NPY_TOSTRING(NAME),          \
            simd_arg_converter, &arg1,                    \
            simd_arg_converter, &arg2,                    \
            simd_arg_converter, &arg3,                    \
            simd_arg_converter, &arg4                     \
        )) return NULL;                                   \
        simd_data data = {.RET = npyv_##NAME(             \
            arg1.data.IN0, arg2.data.IN1,                 \
            arg3.data.IN2, arg4.data.IN3                  \
        )};                                               \
        simd_arg_free(&arg1);                             \
        simd_arg_free(&arg2);                             \
        simd_arg_free(&arg3);                             \
        simd_arg_free(&arg4);                             \
        simd_arg ret = {                                  \
            .data = data, .dtype = simd_data_##RET        \
        };                                                \
        return simd_arg_to_obj(&ret);                     \
    }

#define SIMD_IMPL_INTRIN_8(NAME, RET, IN0, IN1, IN2, IN3, \
                                      IN4, IN5, IN6, IN7) \
    static PyObject *simd__intrin_##NAME                  \
    (PyObject* NPY_UNUSED(self), PyObject *args)          \
    {                                                     \
        simd_arg arg1 = {.dtype = simd_data_##IN0};       \
        simd_arg arg2 = {.dtype = simd_data_##IN1};       \
        simd_arg arg3 = {.dtype = simd_data_##IN2};       \
        simd_arg arg4 = {.dtype = simd_data_##IN3};       \
        simd_arg arg5 = {.dtype = simd_data_##IN4};       \
        simd_arg arg6 = {.dtype = simd_data_##IN5};       \
        simd_arg arg7 = {.dtype = simd_data_##IN6};       \
        simd_arg arg8 = {.dtype = simd_data_##IN7};       \
        if (!PyArg_ParseTuple(                            \
            args, "O&O&O&O&O&O&O&O&:"NPY_TOSTRING(NAME),  \
            simd_arg_converter, &arg1,                    \
            simd_arg_converter, &arg2,                    \
            simd_arg_converter, &arg3,                    \
            simd_arg_converter, &arg4,                    \
            simd_arg_converter, &arg5,                    \
            simd_arg_converter, &arg6,                    \
            simd_arg_converter, &arg7,                    \
            simd_arg_converter, &arg8                     \
        )) return NULL;                                   \
        simd_data data = {.RET = npyv_##NAME(             \
            arg1.data.IN0, arg2.data.IN1,                 \
            arg3.data.IN2, arg4.data.IN3,                 \
            arg5.data.IN4, arg6.data.IN5,                 \
            arg7.data.IN6, arg8.data.IN7                  \
        )};                                               \
        simd_arg_free(&arg1);                             \
        simd_arg_free(&arg2);                             \
        simd_arg_free(&arg3);                             \
        simd_arg_free(&arg4);                             \
        simd_arg_free(&arg5);                             \
        simd_arg_free(&arg6);                             \
        simd_arg_free(&arg7);                             \
        simd_arg_free(&arg8);                             \
        simd_arg ret = {                                  \
            .data = data, .dtype = simd_data_##RET        \
        };                                                \
        return simd_arg_to_obj(&ret);                     \
    }

#define SIMD_IMPL_INTRIN_5(NAME, RET, IN0, IN1, IN2, IN3, IN4) \
    static PyObject *simd__intrin_##NAME                  \
    (PyObject* NPY_UNUSED(self), PyObject *args)          \
    {                                                     \
        simd_arg arg1 = {.dtype = simd_data_##IN0};       \
        simd_arg arg2 = {.dtype = simd_data_##IN1};       \
        simd_arg arg3 = {.dtype = simd_data_##IN2};       \
        simd_arg arg4 = {.dtype = simd_data_##IN3};       \
        simd_arg arg5 = {.dtype = simd_data_##IN4};       \
        if (!PyArg_ParseTuple(                            \
            args, "O&O&O&O&O&:"NPY_TOSTRING(NAME),        \
            simd_arg_converter, &arg1,                    \
            simd_arg_converter, &arg2,                    \
            simd_arg_converter, &arg3,                    \
            simd_arg_converter, &arg4,                    \
            simd_arg_converter, &arg5                     \
        )) return NULL;                                   \
        simd_data data = {.RET = npyv_##NAME(             \
            arg1.data.IN0, arg2.data.IN1,                 \
            arg3.data.IN2, arg4.data.IN3,                 \
            arg5.data.IN4                                 \
        )};                                               \
        simd_arg_free(&arg1);                             \
        simd_arg_free(&arg2);                             \
        simd_arg_free(&arg3);                             \
        simd_arg_free(&arg4);                             \
        simd_arg_free(&arg5);                             \
        simd_arg ret = {                                  \
            .data = data, .dtype = simd_data_##RET        \
        };                                                \
        return simd_arg_to_obj(&ret);                     \
    }

/**
 * Helper macros for repeating and expand a certain macro.
 * Mainly used for converting a scalar to an immediate constant.
 */
#define SIMD__IMPL_COUNT_7(FN, ...)      \
    NPY_EXPAND(FN(0,  __VA_ARGS__))      \
    SIMD__IMPL_COUNT_7_(FN, __VA_ARGS__)

#define SIMD__IMPL_COUNT_8(FN, ...)      \
    SIMD__IMPL_COUNT_7_(FN, __VA_ARGS__) \
    NPY_EXPAND(FN(8,  __VA_ARGS__))

#define SIMD__IMPL_COUNT_15(FN, ...)     \
    NPY_EXPAND(FN(0,  __VA_ARGS__))      \
    SIMD__IMPL_COUNT_15_(FN, __VA_ARGS__)

#define SIMD__IMPL_COUNT_16(FN, ...)      \
    SIMD__IMPL_COUNT_15_(FN, __VA_ARGS__) \
    NPY_EXPAND(FN(16,  __VA_ARGS__))

#define SIMD__IMPL_COUNT_31(FN, ...)     \
    NPY_EXPAND(FN(0,  __VA_ARGS__))      \
    SIMD__IMPL_COUNT_31_(FN, __VA_ARGS__)

#define SIMD__IMPL_COUNT_32(FN, ...)      \
    SIMD__IMPL_COUNT_31_(FN, __VA_ARGS__) \
    NPY_EXPAND(FN(32,  __VA_ARGS__))

#define SIMD__IMPL_COUNT_47(FN, ...)     \
    NPY_EXPAND(FN(0,  __VA_ARGS__))      \
    SIMD__IMPL_COUNT_47_(FN, __VA_ARGS__)

#define SIMD__IMPL_COUNT_48(FN, ...)      \
    SIMD__IMPL_COUNT_47_(FN, __VA_ARGS__) \
    NPY_EXPAND(FN(48,  __VA_ARGS__))

#define SIMD__IMPL_COUNT_63(FN, ...)     \
    NPY_EXPAND(FN(0,  __VA_ARGS__))      \
    SIMD__IMPL_COUNT_63_(FN, __VA_ARGS__)

#define SIMD__IMPL_COUNT_64(FN, ...)      \
    SIMD__IMPL_COUNT_63_(FN, __VA_ARGS__) \
    NPY_EXPAND(FN(64,  __VA_ARGS__))

#define SIMD__IMPL_COUNT_7_(FN, ...)                                \
                                    NPY_EXPAND(FN(1,  __VA_ARGS__)) \
    NPY_EXPAND(FN(2,  __VA_ARGS__)) NPY_EXPAND(FN(3,  __VA_ARGS__)) \
    NPY_EXPAND(FN(4,  __VA_ARGS__)) NPY_EXPAND(FN(5,  __VA_ARGS__)) \
    NPY_EXPAND(FN(6,  __VA_ARGS__)) NPY_EXPAND(FN(7,  __VA_ARGS__))

#define SIMD__IMPL_COUNT_15_(FN, ...)                               \
    SIMD__IMPL_COUNT_7_(FN, __VA_ARGS__)                            \
    NPY_EXPAND(FN(8,  __VA_ARGS__)) NPY_EXPAND(FN(9,  __VA_ARGS__)) \
    NPY_EXPAND(FN(10, __VA_ARGS__)) NPY_EXPAND(FN(11, __VA_ARGS__)) \
    NPY_EXPAND(FN(12, __VA_ARGS__)) NPY_EXPAND(FN(13, __VA_ARGS__)) \
    NPY_EXPAND(FN(14, __VA_ARGS__)) NPY_EXPAND(FN(15, __VA_ARGS__))

#define SIMD__IMPL_COUNT_31_(FN, ...)                               \
    SIMD__IMPL_COUNT_15_(FN, __VA_ARGS__)                           \
    NPY_EXPAND(FN(16, __VA_ARGS__)) NPY_EXPAND(FN(17, __VA_ARGS__)) \
    NPY_EXPAND(FN(18, __VA_ARGS__)) NPY_EXPAND(FN(19, __VA_ARGS__)) \
    NPY_EXPAND(FN(20, __VA_ARGS__)) NPY_EXPAND(FN(21, __VA_ARGS__)) \
    NPY_EXPAND(FN(22, __VA_ARGS__)) NPY_EXPAND(FN(23, __VA_ARGS__)) \
    NPY_EXPAND(FN(24, __VA_ARGS__)) NPY_EXPAND(FN(25, __VA_ARGS__)) \
    NPY_EXPAND(FN(26, __VA_ARGS__)) NPY_EXPAND(FN(27, __VA_ARGS__)) \
    NPY_EXPAND(FN(28, __VA_ARGS__)) NPY_EXPAND(FN(29, __VA_ARGS__)) \
    NPY_EXPAND(FN(30, __VA_ARGS__)) NPY_EXPAND(FN(31, __VA_ARGS__))

#define SIMD__IMPL_COUNT_47_(FN, ...)                               \
    SIMD__IMPL_COUNT_31_(FN, __VA_ARGS__)                           \
    NPY_EXPAND(FN(32, __VA_ARGS__)) NPY_EXPAND(FN(33, __VA_ARGS__)) \
    NPY_EXPAND(FN(34, __VA_ARGS__)) NPY_EXPAND(FN(35, __VA_ARGS__)) \
    NPY_EXPAND(FN(36, __VA_ARGS__)) NPY_EXPAND(FN(37, __VA_ARGS__)) \
    NPY_EXPAND(FN(38, __VA_ARGS__)) NPY_EXPAND(FN(39, __VA_ARGS__)) \
    NPY_EXPAND(FN(40, __VA_ARGS__)) NPY_EXPAND(FN(41, __VA_ARGS__)) \
    NPY_EXPAND(FN(42, __VA_ARGS__)) NPY_EXPAND(FN(43, __VA_ARGS__)) \
    NPY_EXPAND(FN(44, __VA_ARGS__)) NPY_EXPAND(FN(45, __VA_ARGS__)) \
    NPY_EXPAND(FN(46, __VA_ARGS__)) NPY_EXPAND(FN(47, __VA_ARGS__))

#define SIMD__IMPL_COUNT_63_(FN, ...)                               \
    SIMD__IMPL_COUNT_47_(FN, __VA_ARGS__)                           \
    NPY_EXPAND(FN(48, __VA_ARGS__)) NPY_EXPAND(FN(49, __VA_ARGS__)) \
    NPY_EXPAND(FN(50, __VA_ARGS__)) NPY_EXPAND(FN(51, __VA_ARGS__)) \
    NPY_EXPAND(FN(52, __VA_ARGS__)) NPY_EXPAND(FN(53, __VA_ARGS__)) \
    NPY_EXPAND(FN(54, __VA_ARGS__)) NPY_EXPAND(FN(55, __VA_ARGS__)) \
    NPY_EXPAND(FN(56, __VA_ARGS__)) NPY_EXPAND(FN(57, __VA_ARGS__)) \
    NPY_EXPAND(FN(58, __VA_ARGS__)) NPY_EXPAND(FN(59, __VA_ARGS__)) \
    NPY_EXPAND(FN(60, __VA_ARGS__)) NPY_EXPAND(FN(61, __VA_ARGS__)) \
    NPY_EXPAND(FN(62, __VA_ARGS__)) NPY_EXPAND(FN(63, __VA_ARGS__))
