#ifndef NUMPY_CORE_SRC_COMMON_SIMD_TEST_INTRINSICS_H_
#define NUMPY_CORE_SRC_COMMON_SIMD_TEST_INTRINSICS_H_

#include "datatypes.hpp"

namespace np::NPY_CPU_DISPATCH_CURFX(simd_test) {
#if NPY_SIMD

using OverloadCB = PyObject*(*)(const char *, Bytes*);
using GetOverloadCB = OverloadCB*(*)(const uint16_t*, size_t);

template <typename T, typename ...Args>
struct Overload;

template <typename Ret, typename ...Args>
struct Overload<Ret(&)(Args...)> : Overload<Ret, Args...>
{};
template <typename Ret, typename ...Args>
struct Overload<Ret(*)(Args...)> : Overload<Ret, Args...>
{};

template <typename Ret, typename ...Args>
struct Overload {
    using FuncPtr = Ret(*)(Args...);
    static constexpr auto kNumArgs = sizeof...(Args);
    static constexpr auto kArgInd = std::make_index_sequence<kNumArgs>{};

    static PyObject *SignatureIDs()
    {
        uint16_t ids[] = {kGetID<Ret>, kGetID<Args>...};
        return PyByteArray_FromStringAndSize(
            reinterpret_cast<const char *>(ids),
            (kNumArgs + 1) * sizeof(uint16_t)
        );
    }

    static bool TestSignature(uint16_t *ids, size_t len)
    {
        if (len != kNumArgs) {
            return false;
        }
        if constexpr (kNumArgs > 0) {
            return TestSignature_(ids, len, kArgInd);
        }
        return true;
    }

    template <FuncPtr Intrinsic>
    static PyObject* Callback(const char *intrin_name, Bytes *args)
    {
        return Callback_<Intrinsic>(intrin_name, args, kArgInd);
    }

    template <FuncPtr Intrinsic, size_t ...Ind>
    static PyObject *Callback_(const char *intrin_name, Bytes *args,
                               std::index_sequence<Ind...>)
    {
        std::tuple params = {DataType<Args>(args[Ind])...};
        if constexpr (sizeof...(Args) > 0) {
            bool is_null = ((std::get<Ind>(params).IsNull()) || ...);
            if (is_null) {
                return nullptr;
            }
        }
        if constexpr (!std::is_void_v<Ret>) {
            DataType<Ret> r(Intrinsic(std::get<Ind>(params)...));
            return r.ToBytes();
        }
        else {
            Intrinsic(std::get<Ind>(params)...);
            Py_RETURN_NONE;
        }
    }

    template <size_t ...Ind>
    static bool TestSignature_(uint16_t *ids, size_t len,
                              std::index_sequence<Ind...>)
    {
        constexpr uint16_t cids[] = {kGetID<Args>...};
        return ((cids[Ind] == ids[Ind]) && ...);
    }
};


template <size_t num_overloads>
struct IntrinsicObject {
    PyObject_HEAD
    const char *intrin_name;
    OverloadCB intrin_funcs[num_overloads];

    template <typename ...FuncOverload>
    IntrinsicObject(const char *name, FuncOverload ...ptrs)
        : intrin_name(name), intrin_funcs{ptrs...}
    {}
};

template <typename ...Overloads>
class IntrinsicType {
  public:
    static PyTypeObject *GetTypeObject()
    {
        static IntrinsicType type;
        return &type.type_;
    }

  private:
    static constexpr int kNumOverloads = sizeof...(Overloads);
    using IntrinObj = IntrinsicObject<kNumOverloads>;

    IntrinsicType()
    { PyType_Ready(&type_); }

    static PyObject *Call_(PyObject *self_, PyObject *args, PyObject*)
    {
        IntrinObj *self = reinterpret_cast<IntrinObj*>(self_);
        if (!PyTuple_Check(args)) {
            return nullptr;
        }

        PyObject **items = PySequence_Fast_ITEMS(args);
        Py_ssize_t length = PySequence_Fast_GET_SIZE(args);

        constexpr size_t kMaxArgs = 255;
        Bytes data[kMaxArgs];
        uint16_t test_ids[kMaxArgs];

        bool(*test_signatures[])(uint16_t*, size_t) = {
            Overloads::TestSignature...
        };
        for (Py_ssize_t i = 0; i < length; ++i) {
            PyObject *obj = items[i];
            if (!PyByteArray_Check(obj)) {
                goto err;
            }
            auto d = Bytes(reinterpret_cast<PyByteArrayObject*>(obj));
            if (d.IsNull()) {
                return nullptr;
            }
            test_ids[i] = d.ID();
            data[i] = d;
        }
        for (size_t i = 0; i < kNumOverloads; ++i) {
            auto test_func = test_signatures[i];
            bool test = test_func(test_ids, static_cast<size_t>(length));
            if (!test) {
                continue;
            }
            auto f = self->intrin_funcs[i];
            return f(self->intrin_name, data);
        }
    err:
        PyErr_Format(PyExc_TypeError, "no matching signature for call %s()", self->intrin_name);
        return nullptr;
    }

    static PyObject *Dict_()
    {
        PyObject *dict = PyDict_New();
        PyObject *signatures = PyTuple_New(kNumOverloads);
        PyObject *sig_objs[] = {Overloads::SignatureIDs()...};

        if (dict == nullptr || signatures == nullptr) {
            goto clear;
        }
        for (Py_ssize_t i = 0; i < kNumOverloads; ++i) {
            if (sig_objs[i] == nullptr) {
                goto clear;
            }
            PyTuple_SET_ITEM(signatures, i, sig_objs[i]);
            sig_objs[i] = nullptr;
        }
        if (PyDict_SetItemString(dict, "signatures", signatures) < 0) {
            goto clear;
        }
        return dict;
    clear:
        for (Py_ssize_t i = 0; i < kNumOverloads; ++i) {
            if (sig_objs[i] != nullptr) {
                Py_DECREF(sig_objs[i]);
            }
        }
        Py_DECREF(signatures);
        Py_DECREF(dict);
        return nullptr;
    }

    PyTypeObject type_ = {
        PyVarObject_HEAD_INIT(&PyType_Type, 0)
        "Intrinsic", /* tp_name */
        sizeof(IntrinObj), /* tp_basicsize */
        0, /* tp_itemsize */
        0, /* tp_dealloc */
        0, /* tp_print */
        0, /* tp_getattr */
        0, /* tp_setattr */
        0, /* tp_reserved */
        0, /* tp_repr */
        0, /* tp_as_number */
        0, /* tp_as_sequence */
        0, /* tp_as_mapping */
        0, /* tp_hash */
        Call_,   /* tp_call */
        0,       /* tp_str */
        0,       /* tp_getattro */
        0,       /* tp_setattro */
        0,       /* tp_as_buffer */
        0,       /* tp_flags */
        0,       /* tp_doc */
        0,       /* tp_traverse */
        0,       /* tp_clear */
        0,       /* tp_richcompare */
        0,       /* tp_weaklistoffset */
        0,       /* tp_iter */
        0,       /* tp_iternext */
        0,       /* tp_methods */
        0,       /* tp_members */
        0,       /* tp_getset */
        0,       /* tp_base */
        Dict_()  /* tp_dict */
    };
};

template <auto ...FuncOverload>
inline void Intrinsic(PyObject *mod, const char *name)
{
    PyTypeObject *intrinsic_type = IntrinsicType<
        Overload<decltype(FuncOverload)>...
    >::GetTypeObject();
    if (intrinsic_type == nullptr) {
        return;
    }

    using IntrinObj = IntrinsicObject<sizeof...(FuncOverload)>;
    IntrinObj *cfunc = PyObject_New(IntrinObj, intrinsic_type);
    if (cfunc == nullptr) {
        return;
    }
    new(cfunc) IntrinObj(name, Overload<decltype(FuncOverload)>::template Callback<FuncOverload>...);
    if (PyObject_SetAttrString(mod, name, reinterpret_cast<PyObject*>(cfunc)) < 0) {
        Py_DECREF(cfunc);
    }
}

#endif
} // namespace np::simd_test

#endif // NUMPY_CORE_SRC_COMMON_SIMD_TEST_INTRINSICS_H_
