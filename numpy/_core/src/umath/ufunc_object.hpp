#ifndef NUMPY_CORE_SRC_COMMON_UFUNC_OBJECT_HPP
#define NUMPY_CORE_SRC_COMMON_UFUNC_OBJECT_HPP

#ifndef NPY_NO_DEPRECATED_API
    #define NPY_NO_DEPRECATED_API NPY_API_VERSION
#endif

#ifndef NO_IMPORT_ARRAY
    #define NO_IMPORT_ARRAY // supress warning: ‘int _import_array()’ defined but not used [-Wunused-function]
#endif

#ifndef _UMATHMODULE
    #define _UMATHMODULE // initlize UFunc API for internal build
#endif

#include <numpy/ndarrayobject.h>
#include <numpy/ufuncobject.h>

#include "common.hpp"

#include <functional>

namespace np {

namespace ufunc_helper {

template <typename T>
struct IsSliceBase_ : std::false_type {};
template <typename T>
struct IsSliceBase_<Slice<T>> : std::true_type {};
// Determines if a type is a specialization of Slice.
template <typename T, typename TBase=std::remove_cv_t<std::remove_reference_t<T>>>
constexpr bool kIsSliceBase = IsSliceBase_<TBase>::value;

template <typename T>
struct FuncDeduce_;
template <typename Ret, typename ..._Args>
struct FuncDeduce_<std::function<Ret(_Args...)>> {
    static constexpr int kNumArgs = sizeof...(_Args);
    static constexpr int kNumIn = ((
        std::is_const_v<std::remove_reference_t<_Args>> ||
        !std::is_reference_v<_Args>
    ) + ...);
    static constexpr int kNumOut = kNumArgs - kNumIn;
    static constexpr bool kIsSlice = (kIsSliceBase<_Args> && ...);
    using Args = std::tuple<_Args...>;
};
// This metafunction deduces information about a callable object, including the number of input
// and output arguments, whether any of the input arguments are slices, and the types of all the arguments.
template<typename T>
struct FuncDeduce : FuncDeduce_<decltype(std::function{std::declval<T>()})> {};
// Determines the type ID of a Slice<T>'s element type.
template <typename T, typename TBase=std::remove_cv_t<std::remove_reference_t<T>>>
constexpr char ElementTypeID = static_cast<char>(kTypeID<typename TBase::ElementType>);

} // namespace np::ufunc_helper

/** Universal function object.
 *
 * This class simplifies the creation of UFunc objects by deducing
 * the signature of function pointers in order to determine the desired operands of
 * the overloaded functions (inner loops) rather than specify it manually similar to C API.
 * Since this class still counts on the C API, the following limitations are observed:
 *   - All C++ functions must have the same number of inputs and outputs
 *   - Outputs parmaters must start at the end of the inputs.
 *   - No support for return type.
 *   - Only datatype Slice<T> is supported and there's no room for
 *     supporting scalars yet.
 *
 * Some examples:
 *
 * @code
 * template<typename T>
 * static void Add(
 *      // Prpamerts without references represent the inputs
 *      np::Slice<T> in0, np::Slice<T> in1,
 *      // and with reference `&` represents the output
 *      np::Slice<T> &out
 * )
 * {
 *     for (auto a = in0.begin(), b = in1.begin(), c = out.begin(),
 *             end = out.end();  c != end; ++a, ++b, ++c) {
 *         *c = *a + *b;
 *     }
 * }
 *
 * template<typename T>
 * static void CompareEqual(np::Slice<T> in0, np::Slice<T> in1,
 *      // non-symmetric types are allowed
 *      np::Slice<Bool> &out
 * )
 * {
 *     auto c = out.begin();
 *     auto end = c.end();
 *     for (auto a = in0.begin(), b = in1.begin(), c != end; ++a, ++b, ++c) {
 *         *c = *a == *b;
 *     }
 * }
 *
 * void MyCallee(PyObject *module_dict)
 * {
 *     using namespace np;
 *     static UFuncObject add_obj(
 *          // name/doc of the ufunc
 *          "add", "this doc of add function",
 *          // passing the pointers of the inner functions
 *          // Signature of these functions will be detected and
 *          // converted into python calls.
 *          Add<Float>, Add<Double> // more data types are welcome
 *     );
 *     if (!add_obj.IsNull()) {
 *        // now the following signatures are supported from the python level:
 *        // add(np.array(dtype=np.float32), np.array(dtype=np.float32), out=np.array(dtype=np.float32))
 *        // add(np.array(dtype=np.float64), np.array(dtype=np.float64), out=np.array(dtype=np.float64))
 *        PyDict_SetItemString(module_dict, "add", add_obj);
 *        Py_DECREF(add_obj);
 *     }
 *     static UFuncObject compare_equal_obj(
 *          "compare_equal", "this doc of compare_equal function",
 *          // passing the pointers of the inner functions
 *          // Signature of these functions will be detected and
 *          // converted into python calls.
 *          Add<Float>, Add<Double> // more data types are welcome
 *     );
 *     if (!compare_equal_obj.IsNull()) {
 *         // now the following signatures are supported from the python level:
 *         // add(np.array(dtype=np.float32), np.array(dtype=np.float32), out=np.array(dtype=np.bool_))
 *         // add(np.array(dtype=np.float64), np.array(dtype=np.float64), out=np.array(dtype=np.bool_))
 *     }
 * }
 * @endcode
 */
template <typename ...TFuncOverloads>
class UFuncObject {
  public:
    /// Number of overloaded functions (inner loops)
    static constexpr int kNumOverloads = sizeof...(TFuncOverloads);
    static_assert(kNumOverloads > 0, "Requires at least one function pointer");
    static_assert(
        ((std::is_function_v<std::remove_pointer_t<TFuncOverloads>>) && ...),
        "Expected function pointers"
    );
   /** Constructs a UFuncObject
    * @param name           The name of the UFuncObject
    * @param doc            The documentation of the UFuncObject
    * @param identity       The identity of the UFuncObject
    * @param identity_value The identity value of the UFuncObject
    * @param signature      The signature of the UFuncObject
    * @param funcs          The function pointers of each overload.
    */
    constexpr UFuncObject(const char *name, const char *doc,
                         int identity, PyObject *identity_value,
                         const char *signature, TFuncOverloads ...funcs)
        : UFuncObject(
            name, doc, identity, identity_value, signature,
            funcs...,
            std::make_index_sequence<std::tuple_size_v<AllParamsTypes_>>{})
    {}
   /** Constructs a UFuncObject
    * @param name           The name of the UFuncObject
    * @param doc            The documentation of the UFuncObject
    * @param funcs          The function pointers of each overload.
    */
    constexpr UFuncObject(const char *name, const char *doc, TFuncOverloads ...funcs)
        : UFuncObject(
            name, doc, PyUFunc_None, nullptr, nullptr,
            funcs...)
    {}
    /// Returns false if the construction fails.
    /// Since there's no support of exceptions, this function must be called
    /// after construction.
    bool IsNull() const
    { return ref_ == nullptr; }
    /// Update an existance overload of UFuncObject.
    template <typename TFunc>
    constexpr void Update(TFunc func)
    {
        constexpr size_t idx = Index_<TFunc>(
            std::make_index_sequence<kNumOverloads>{}
        );
        printf("%d\n", idx);
        data_[idx] = reinterpret_cast<void*>(func);
    }
    /// Implicitly converts the UFuncObject to a PyObject pointer.
    operator PyObject*()
    { return ref_; }
    /// Implicitly converts the UFuncObject to a PyUFuncObject pointer.
    operator PyUFuncObject*()
    { return reinterpret_cast<PyUFuncObject*>(ref_); }

  private:
    // get the first function signature
    using FuncDeduce_ = ufunc_helper::FuncDeduce<std::tuple_element_t<
        0, std::tuple<TFuncOverloads...>
    >>;
    // number of ufunc args
    static constexpr int kNumArgs_ = FuncDeduce_::kNumArgs;
    // number of inputs
    static constexpr int kNumIn_ = FuncDeduce_::kNumIn;
    // number of ouput
    static constexpr int kNumOut_ = FuncDeduce_::kNumOut;
    static_assert(
        kNumOut_ > 0,
        "Overloaded functions must have at least one return (reference)"
    );
    static_assert(
        ((ufunc_helper::FuncDeduce<TFuncOverloads>::kNumIn == kNumIn_) && ...),
        "Overloaded functions must have the same number of inputs,"
        "variant order overloading is not supported yet"
    );
    static_assert(
        ((ufunc_helper::FuncDeduce<TFuncOverloads>::kNumOut == kNumOut_) && ...),
        "Overloaded functions must have the same number of output,"
        "variant order overloading is not supported yet"
    );
    static_assert(
        (ufunc_helper::FuncDeduce<TFuncOverloads>::kIsSlice && ...),
        "Only np::Slice<T> is supported as parmater type at current moment"
    );

    // concate all types of all functions parmaters into a single tuple
    using AllParamsTypes_ = decltype(std::tuple_cat(
        std::declval<typename ufunc_helper::FuncDeduce<TFuncOverloads>::Args>()...
    ));

    // the main constracing function, requires the indices
    // of `AllParamsTypes_` so we keep it private.
    template <std::size_t ...Ind>
    constexpr UFuncObject(const char *name, const char *doc,
                          int identity, PyObject *identity_value,
                          const char *signature, TFuncOverloads ...funcs,
                          std::index_sequence<Ind...>
    )
        : // intilize C function pointers
          functions_{ToLegacyCFunc_(std::forward<TFuncOverloads>(funcs))...},
          // convert all parmaters types into TypeIDs
          ids_{ufunc_helper::ElementTypeID<
              std::tuple_element_t<Ind, AllParamsTypes_>
          >...},
          // Initlize the C++ function pointers
          data_{reinterpret_cast<void*>(funcs)...},
          ref_(
              PyUFunc_FromFuncAndDataAndSignatureAndIdentity(
                  functions_, data_, ids_, kNumOverloads, kNumIn_, kNumOut_, identity,
                  name, doc, 0/*unused*/, signature, identity_value
              )
          )
    {}

    template <typename TFunc, size_t ...Ind>
    static constexpr size_t Index_(std::index_sequence<Ind...>)
    {
        static_assert(
            (std::is_same_v<TFuncOverloads, TFunc> || ...),
            "Unable to find an existance signature for this overload"
        );
        return ((std::is_same_v<TFuncOverloads, TFunc> ? Ind + 1 : 0) + ...) - 1;
    }

    // returns C function pointer after wraps C++ calls, we use
    // data* to store C++ function pointer
    template <typename TFunc, typename TArgs, size_t ...Ind>
    static constexpr PyUFuncGenericFunction ToLegacyCFunc_(std::index_sequence<Ind...>)
    {
        return [](char **args, IntPtr const *dimensions, IntPtr const *strides, void *data) -> void {
            using base_types = std::tuple<std::remove_reference_t<
                std::tuple_element_t<Ind, TArgs>
            >...>;
            auto params = base_types(
                std::tuple_element_t<Ind, base_types>(
                    args[Ind], dimensions[0], strides[Ind]
                )...
            );
            std::apply(reinterpret_cast<TFunc>(data), params);
        };
    }
    template <typename TFunc, typename TDeduce = ufunc_helper::FuncDeduce<TFunc>>
    static constexpr PyUFuncGenericFunction ToLegacyCFunc_(TFunc)
    {
        return ToLegacyCFunc_<TFunc, typename TDeduce::Args>(
            std::make_index_sequence<TDeduce::kNumArgs>{}
        );
    }

    PyUFuncGenericFunction functions_[kNumOverloads];
    char ids_[kNumArgs_*kNumOverloads];
    void *data_[kNumOverloads];
    PyObject *ref_;
};
} // namespace np
#endif // NUMPY_CORE_SRC_COMMON_UFUNC_OBJECT_HPP

