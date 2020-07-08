#if !NPY_SIMD
    #error "Not a standalone header, only works through 'simd.dispatch.c.src'"
#endif

#define SIMD_INTRIN_DEF(NAME) \
    { NPY_TOSTRING(NAME), simd__intrin_##NAME, METH_VARARGS, NULL } , // comma

static int simd__no_arguments(PyObject *args, const char* method_name)
{
    if (args == NULL) {
        return 0;
    }
    assert(PyTuple_Check(args));
    Py_ssize_t obj_arg_len = PyTuple_GET_SIZE(args);
    if (obj_arg_len != 0) {
        PyErr_Format(PyExc_RuntimeError,
            "%s(), takes no arguments, given(%d)", method_name, obj_arg_len
        );
        return -1;
    }
    return 0;
}

#define SIMD_IMPL_INTRIN_0(NAME, RET)                     \
    static PyObject *simd__intrin_##NAME                  \
    (PyObject* NPY_UNUSED(self), PyObject *args)          \
    {                                                     \
        if (simd__no_arguments(                           \
            args, NPY_TOSTRING(NAME)                      \
        )) return NULL;                                   \
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
        if (simd__no_arguments(                           \
            args, NPY_TOSTRING(NAME)                      \
        )) return NULL;                                   \
        npyv_##NAME();                                    \
        Py_RETURN_NONE;                                   \
    }

#define SIMD_IMPL_INTRIN_1(NAME, RET, IN0)                \
    static PyObject *simd__intrin_##NAME                  \
    (PyObject* NPY_UNUSED(self), PyObject *args)          \
    {                                                     \
        simd_arg req_args[] = {                           \
            {.dtype = simd_data_##IN0},                   \
        };                                                \
        if (simd_args_from_tuple(                         \
            args, req_args, 1, NPY_TOSTRING(NAME))        \
        ) return NULL;                                    \
        simd_data r = {.RET = npyv_##NAME(                \
            req_args[0].data.IN0                          \
        )};                                               \
        simd_args_sequence_free(req_args, 1);             \
        req_args[0].data = r;                             \
        req_args[0].dtype = simd_data_##RET;              \
        return simd_arg_to_obj(req_args);                 \
    }

#define SIMD_IMPL_INTRIN_2(NAME, RET, IN0, IN1)           \
    static PyObject *simd__intrin_##NAME                  \
    (PyObject* NPY_UNUSED(self), PyObject *args)          \
    {                                                     \
        simd_arg req_args[] = {                           \
            {.dtype = simd_data_##IN0},                   \
            {.dtype = simd_data_##IN1},                   \
        };                                                \
        if (simd_args_from_tuple(                         \
            args, req_args, 2, NPY_TOSTRING(NAME))        \
        ) {                                               \
            return NULL;                                  \
        }                                                 \
        simd_data r = {.RET = npyv_##NAME(                \
            req_args[0].data.IN0,                         \
            req_args[1].data.IN1                          \
        )};                                               \
        simd_args_sequence_free(req_args, 2);             \
        req_args[0].data = r;                             \
        req_args[0].dtype = simd_data_##RET;              \
        return simd_arg_to_obj(req_args);                 \
    }

#define SIMD__REPEAT_2IMM(C, NAME, IN0) \
    C == req_args[1].data.u8 ? NPY_CAT(npyv_, NAME)(req_args[0].data.IN0, C) :

#define SIMD_IMPL_INTRIN_2IMM(NAME, RET, IN0, CONST_RNG)  \
    static PyObject *simd__intrin_##NAME                  \
    (PyObject* NPY_UNUSED(self), PyObject *args)          \
    {                                                     \
        simd_arg req_args[] = {                           \
            {.dtype = simd_data_##IN0},                   \
            {.dtype = simd_data_u8},                      \
        };                                                \
        if (simd_args_from_tuple(                         \
            args, req_args, 2, NPY_TOSTRING(NAME))        \
        ) {                                               \
            return NULL;                                  \
        }                                                 \
        simd_data r;                                      \
        r.RET = NPY_CAT(SIMD__IMPL_COUNT_, CONST_RNG)(    \
            SIMD__REPEAT_2IMM, NAME, IN0                  \
        ) npyv_##NAME(req_args[0].data.IN0, 0);           \
        simd_args_sequence_free(req_args, 2);             \
        req_args[0].data = r;                             \
        req_args[0].dtype = simd_data_##RET;              \
        return simd_arg_to_obj(req_args);                 \
    }

#define SIMD_IMPL_INTRIN_3(NAME, RET, IN0, IN1, IN2)      \
    static PyObject *simd__intrin_##NAME                  \
    (PyObject* NPY_UNUSED(self), PyObject *args)          \
    {                                                     \
        simd_arg req_args[] = {                           \
            {.dtype = simd_data_##IN0},                   \
            {.dtype = simd_data_##IN1},                   \
            {.dtype = simd_data_##IN2},                   \
        };                                                \
        if (simd_args_from_tuple(                         \
            args, req_args, 3, NPY_TOSTRING(NAME))        \
        ) {                                               \
            return NULL;                                  \
        }                                                 \
        simd_data r = {.RET = npyv_##NAME(                \
            req_args[0].data.IN0,                         \
            req_args[1].data.IN1,                         \
            req_args[2].data.IN2                          \
        )};                                               \
        simd_args_sequence_free(req_args, 3);             \
        req_args[0].data = r;                             \
        req_args[0].dtype = simd_data_##RET;              \
        return simd_arg_to_obj(req_args);                 \
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
