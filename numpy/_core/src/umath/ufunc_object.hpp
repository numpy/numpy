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
struct ObjFuncDeduce_;

template<typename Ret, typename Class, typename ..._Args>
struct ObjFuncDeduce_<Ret(Class::*)(_Args...)>
{
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
struct ObjFuncDeduce : ObjFuncDeduce_<decltype(&T::operator())> {};
// Determines the type ID of a Slice<T>'s element type.
template <typename T, typename TBase=std::remove_cv_t<std::remove_reference_t<T>>>
constexpr char ElementTypeID = static_cast<char>(kTypeID<typename TBase::ElementType>);

} // namespace np::ufunc_helper

/** Universal Function Object Wrapper for C UFunc.
 *
 * This class provides a convenient way to create universal function (UFunc) objects in C++ for use with NumPy internaly,
 * it leverages template metaprogramming to deduce function signatures and manage the conversion of C++ functor objects
 * to be compatible with NumPy UFuncs.
 *
 * Constraints:
 * - All functor types must have a uniform number of input and output parameters.
 * - Output parameters must be passed by reference and follow all input parameters in the functor's signature.
 * - Direct return values are not supported; use output parameters to return results.
 * - Only supports parameters of type `np::Slice<T>`, where `T` can be any data type compatible with NumPy arrays.
 * - The class does not manage the lifetime of the created PyObject*; users are responsible for reference counting.
 *
 ** Usage Examples:
 *
 * Define functor objects that implement the `operator()` method with desired functionality. Inputs are
 * passed by value, and outputs are passed by non-const reference. These structs can then be used to instantiate
 * `UFuncObject`, which automatically makes these operations available as UFuncs.
 *
 * @code
 * template<typename T>
 * struct AddOP {
 *    void operator ()(
 *      // Prpamerts without references represent the inputs
 *      Slice<T> in0, Slice<T> in1,
 *      // and with reference `&` represents the output
 *      np::Slice<T> &out
 *    )
 *    {
 *        for (auto a = in0.Begin(), b = in1.Begin(), c = out.Begin(),
 *             end = out.End();  c != end; ++a, ++b, ++c) {
 *            *c = *a + *b;
 *        }
 *    }
 * };
 *
 * template<typename T>
 * struct CompareEqualOP {
 *    void operator ()(
 *      np::Slice<T> in0, np::Slice<T> in1,
 *      // non-symmetric types are allowed
 *      np::Slice<Bool> &out
 *    )
 *    {
 *        auto c = out.Begin();
 *        auto end = c.End();
 *        for (auto a = in0.Begin(), b = in1.Begin(), c != end; ++a, ++b, ++c) {
 *            *c = *a == *b;
 *        }
 *    }
 * };
 *
 * void MyCallee(PyObject *module_dict)
 * {
 *     using namespace np;
 *     // Instantiate and register the 'add' UFunc
 *     static UFuncObject<AddOP<Float>, AddOP<Double>> add_obj("add", "Addition operation");
 *     // Register the UFunc with the Python module if not null
 *     if (!add_obj.IsNull()) {
 *        // now the following signatures are supported from the python level:
 *        // add(np.array(dtype=np.float32), np.array(dtype=np.float32), out=np.array(dtype=np.float32))
 *        // add(np.array(dtype=np.float64), np.array(dtype=np.float64), out=np.array(dtype=np.float64))
 *        PyDict_SetItemString(module_dict, "add", add_obj);
 *        Py_DECREF(add_obj);
 *     }
 *     static UFuncObject <
 *         // passing the pointers of the inner functions
 *         // Signature of these functions will be detected and
 *         // converted into python calls.
 *         CompareEqualOP<Float>, CompareEqualOP<Double> // more data types are welcome
 *     > compare_equal_obj(
 *          "compare_equal", "this doc of compare_equal function",
 *     );
 *     if (!compare_equal_obj.IsNull()) {
 *        // now the following signatures are supported from the python level:
 *        // compare_equal(np.array(dtype=np.float32), np.array(dtype=np.float32), out=np.array(dtype=np.bool_))
 *        // compare_equal(np.array(dtype=np.float64), np.array(dtype=np.float64), out=np.array(dtype=np.bool_))
 *        PyDict_SetItemString(module_dict, "compare_equal", compare_equal_obj);
 *        Py_DECREF(add_obj);
 *     }
 * }
 * @endcode
 *
 * @tparam TObjFuncOverloads Variadic template parameters representing the functor types that define the operations.
 */
template <typename ...TObjFuncOverloads>
class UFuncObject {
  public:
    /// Number of overloaded functions (inner loops)
    static constexpr int kNumOverloads = sizeof...(TObjFuncOverloads);
    /** Constructs a UFuncObject with detailed configuration.
     * @param name The name of the UFunc, as it will appear in Python.
     * @param doc Documentation string for the UFunc, explaining its purpose and usage.
     * @param identity Identity element for the UFunc operation, used in reduction operations.
     * @param identity_value The identity value, typically a PyObject representing the identity.
     * @param signature The signature string for the UFunc, describing input and output types.
     */
    constexpr UFuncObject(const char *name, const char *doc,
                         int identity, PyObject *identity_value,
                         const char *signature)
        : UFuncObject(
            name, doc, identity, identity_value, signature,
            std::make_index_sequence<std::tuple_size_v<AllParamsTypes_>>{})
    {}
    /** Simplified constructor for UFuncObject without identity and signature.
     * @param name The name of the UFunc.
     * @param doc Documentation string for the UFunc.
     */
    constexpr UFuncObject(const char *name, const char *doc)
        : UFuncObject(
            name, doc, PyUFunc_None, nullptr, nullptr)
    {}
    /// Checks if the UFuncObject was successfully constructed.
    /// @return True if the object is "not" in a valid state, False otherwise.
    bool IsNull() const
    { return ref_ == nullptr; }
    /// Updates an existing signature of the UFuncObject with a new functor type.
    /// @tparam TObjFunc The type of the functor object to update the UFunc with.
    template <typename TObjFunc>
    constexpr void Update()
    {
        constexpr size_t idx = Index_<TObjFunc>(
            std::make_index_sequence<kNumOverloads>{}
        );
        functions_[idx] = ToLegacyCFunc_<TObjFunc>();
    }
    /// Implicitly converts the UFuncObject to a PyObject pointer.
    operator PyObject*()
    { return ref_; }
    /// Implicitly converts the UFuncObject to a PyUFuncObject pointer.
    operator PyUFuncObject*()
    { return reinterpret_cast<PyUFuncObject*>(ref_); }

  private:
    static_assert(kNumOverloads > 0, "UFuncObject requires at least one functor type as a template parameter.");
    // get the first function signature
    using ObjFuncDeduce_ = ufunc_helper::ObjFuncDeduce<std::tuple_element_t<
        0, std::tuple<TObjFuncOverloads...>
    >>;
    // number of ufunc args
    static constexpr int kNumArgs_ = ObjFuncDeduce_::kNumArgs;
    // number of inputs
    static constexpr int kNumIn_ = ObjFuncDeduce_::kNumIn;
    // number of ouput
    static constexpr int kNumOut_ = ObjFuncDeduce_::kNumOut;
    static_assert(
        kNumOut_ > 0,
        "The functor types must have at least one output which can defined by passing reference to parameter type."
    );
    static_assert(
        ((ufunc_helper::ObjFuncDeduce<TObjFuncOverloads>::kNumIn == kNumIn_) && ...),
        "All functor types must have the same number of input parameters."
        "Variant input parameter counts are not supported."
    );
    static_assert(
        ((ufunc_helper::ObjFuncDeduce<TObjFuncOverloads>::kNumOut == kNumOut_) && ...),
        "All functor types must have the same number of output parameters."
        "Variant output parameter counts are not supported."
    );
    static_assert(
        (ufunc_helper::ObjFuncDeduce<TObjFuncOverloads>::kIsSlice && ...),
        "Only np::Slice<T> parameter types are supported."
        "Scalar types or other container types are not currently supported."
    );

    // Concatenates the parameter types of all provided functor objects into a single tuple.
    using AllParamsTypes_ = decltype(std::tuple_cat(
        std::declval<typename ufunc_helper::ObjFuncDeduce<TObjFuncOverloads>::Args>()...
    ));
    // The main constructor for UFuncObject, leveraging template metaprogramming to deduce and manage parameter types.
    // Requires the indices of `AllParamsTypes_` so we keep it private.
    template <std::size_t ...Ind>
    constexpr UFuncObject(const char *name, const char *doc,
                          int identity, PyObject *identity_value,
                          const char *signature, std::index_sequence<Ind...>
    )
        : // intilize C function pointers
          functions_{ToLegacyCFunc_<TObjFuncOverloads>()...},
          // convert all parmaters types into TypeIDs
          ids_{ufunc_helper::ElementTypeID<
              std::tuple_element_t<Ind, AllParamsTypes_>
          >...},
          // Initlize the C++ function pointers
          data_{nullptr},
          ref_(
              PyUFunc_FromFuncAndDataAndSignatureAndIdentity(
                  functions_, data_, ids_, kNumOverloads, kNumIn_, kNumOut_, identity,
                  name, doc, 0/*unused*/, signature, identity_value
              )
          )
    {}
    template <typename TObjFunc, size_t ...Ind>
    static constexpr size_t Index_(std::index_sequence<Ind...>)
    {
        using ObjArgs_ = typename ufunc_helper::ObjFuncDeduce<TObjFunc>::Args;
        static_assert(
            (std::is_same_v<typename ufunc_helper::ObjFuncDeduce<TObjFuncOverloads>::Args,
                            ObjArgs_> || ...),
            "Unable to find a matching signature for this functor type."
            "Ensure the functor's parameter type align with those of the existing functors types."
        );
        return ((std::is_same_v<
                    typename ufunc_helper::ObjFuncDeduce<TObjFuncOverloads>::Args, ObjArgs_
                > ? Ind + 1 : 0) + ...) - 1;
    }

    // returns C function pointer after wraps C++ calls
    template <typename TObjFunc, typename TArgs, size_t ...Ind>
    static constexpr PyUFuncGenericFunction ToLegacyCFunc_(std::index_sequence<Ind...>)
    {
        return [](char **args, IntPtr const *dimensions, IntPtr const *strides, void*) -> void {
            using base_types = std::tuple<std::remove_reference_t<
                std::tuple_element_t<Ind, TArgs>
            >...>;
            auto params = base_types(
                std::tuple_element_t<Ind, base_types>(
                    args[Ind], dimensions[0], strides[Ind]
                )...
            );
            TObjFunc f;
            std::apply(f, params);
        };
    }
    template <typename TObjFunc, typename TDeduce = ufunc_helper::ObjFuncDeduce<TObjFunc>>
    static constexpr PyUFuncGenericFunction ToLegacyCFunc_()
    {
        return ToLegacyCFunc_<TObjFunc, typename TDeduce::Args>(
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

