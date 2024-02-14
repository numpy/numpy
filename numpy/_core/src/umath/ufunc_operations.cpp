#include "ufunc_object.hpp"

#include "ufunc_type_resolution.h"

#include "ufunc_operations.h"
#include "_umath_doc_generated.h"

using namespace np;

template <typename T>
struct OnesLikeOP {
    void operator ()(Slice<T>, Slice<T> &out)
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
};

template <typename T>
struct CopysignOP {
    void operator ()(Slice<T> in0, Slice<T> in1, Slice<T> &out)
    {
        for (auto a = in0.Begin(), b = in1.Begin(), c = out.Begin(),
             end = out.End();  c != end; ++a, ++b, ++c) {
            *c = Copysign(*a, *b);
        }
    }
};

int InitOperations(PyObject *mdict)
{
    // This is no longer used as numpy.ones_like, however it is
    // still used by some internal calls.
    static UFuncObject<
        OnesLikeOP<Bool>,
        OnesLikeOP<Byte>, OnesLikeOP<UByte>,
        OnesLikeOP<Short>, OnesLikeOP<UShort>,
        OnesLikeOP<Int>, OnesLikeOP<UInt>,
        OnesLikeOP<Long>, OnesLikeOP<ULong>,
        OnesLikeOP<LongLong>, OnesLikeOP<ULongLong>,
        OnesLikeOP<Half>, OnesLikeOP<Float>, OnesLikeOP<Double>, OnesLikeOP<LongDouble>,
        OnesLikeOP<CFloat>, OnesLikeOP<CDouble>, OnesLikeOP<CLongDouble>,
        OnesLikeOP<TimeDelta>, OnesLikeOP<DateTime>, OnesLikeOP<Object>
    > ones_like_obj("_ones_like", DOC_NUMPY__CORE_UMATH__ONES_LIKE);

    static_cast<PyUFuncObject*>(ones_like_obj)->type_resolver = &PyUFunc_OnesLikeTypeResolver;
    PyDict_SetItemString(mdict, "_ones_like", ones_like_obj);
    Py_DECREF(ones_like_obj);

    static UFuncObject<
        CopysignOP<Half>, CopysignOP<Float>,
        CopysignOP<Double>, CopysignOP<LongDouble>
    > copysign_obj(
        "copysign", DOC_NUMPY__CORE_UMATH_COPYSIGN
    );
    if (copysign_obj.IsNull()) {
        return -1;
    }
    PyDict_SetItemString(mdict, "copysign", copysign_obj);
    Py_DECREF(copysign_obj);
    return 0;
}
