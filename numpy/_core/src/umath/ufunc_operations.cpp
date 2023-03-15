#include "ufunc_object.hpp"

#include "ufunc_type_resolution.h"

#include "ufunc_operations.h"
#include "_umath_doc_generated.h"

using namespace np;

template <typename T>
static void op_ones_like(Slice<T>, Slice<T> &out)
{
    for (auto &a : out) {
        if constexpr (std::is_same_v<T, Half>) {
            a = Half::FromBits(0x3c00u);
        }
        else {
            a = 1;
        }
    }
}

template <typename T>
static void op_copysign(Slice<T> in0, Slice<T> in1, Slice<T> &out)
{
    for (auto a = in0.Begin(), b = in1.Begin(), c = out.Begin(),
            end = out.End();  c != end; ++a, ++b, ++c) {
        *c = Copysign(*a, *b);
    }
}

int InitOperations(PyObject *mdict)
{
    // This is no longer used as numpy.ones_like, however it is
    // still used by some internal calls.
    static UFuncObject ones_like_obj(
        "_ones_like", DOC_NUMPY_CORE_UMATH__ONES_LIKE,
        op_ones_like<Bool>,
        op_ones_like<Byte>, op_ones_like<UByte>,
        op_ones_like<Short>, op_ones_like<UShort>,
        op_ones_like<Int>, op_ones_like<UInt>,
        op_ones_like<Long>, op_ones_like<ULong>,
        op_ones_like<LongLong>, op_ones_like<ULongLong>,
        op_ones_like<Half>, op_ones_like<Float>, op_ones_like<Double>, op_ones_like<LongDouble>,
        op_ones_like<CFloat>, op_ones_like<CDouble>, op_ones_like<CLongDouble>,
        op_ones_like<TimeDelta>, op_ones_like<DateTime>, op_ones_like<Object>
    );
    if (ones_like_obj.IsNull()) {
        return -1;
    }
    static_cast<PyUFuncObject*>(ones_like_obj)->type_resolver = &PyUFunc_OnesLikeTypeResolver;
    PyDict_SetItemString(mdict, "_ones_like", ones_like_obj);
    Py_DECREF(ones_like_obj);

    static UFuncObject copysign_obj(
        "copysign", DOC_NUMPY_CORE_UMATH_COPYSIGN,
        op_copysign<Half>, op_copysign<Float>,
        op_copysign<Double>, op_copysign<LongDouble>
    );
    if (copysign_obj.IsNull()) {
        return -1;
    }
    PyDict_SetItemString(mdict, "copysign", copysign_obj);
    Py_DECREF(copysign_obj);
    return 0;
}
