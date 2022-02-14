#ifndef NUMPY_CORE_SRC_COMMON_SIMD_WRAPPER_DATATYPES_HPP_
#define NUMPY_CORE_SRC_COMMON_SIMD_WRAPPER_DATATYPES_HPP_

#include "common.hpp"
#include "simd/simd.h"

#if NPY_SIMD

#if defined(NPY_HAVE_SSE2) && !defined(NPY_HAVE_AVX512F)
    #define NPY_SIMD_STRONG_MASK 0
#else
    #define NPY_SIMD_STRONG_MASK 1
#endif

namespace np::NPY_CPU_DISPATCH_CURFX(simd_ext) {
/**@addtogroup cpp_simd_utils Utils
 * @ingroup cpp_simd
 * @{
 */
namespace helper {
template<size_t size>
struct TLaneBySize_;
template<>
struct TLaneBySize_<sizeof(uint8_t)> {
    using UInt = uint8_t;
    using Int = int8_t;
    using FP = void;
};
template<>
struct TLaneBySize_<sizeof(uint16_t)> {
    using UInt = uint16_t;
    using Int = int16_t;
    using FP = void;
};
template<>
struct TLaneBySize_<sizeof(uint32_t)> {
    using UInt = uint32_t;
    using Int = int32_t;
    using FP = float;
};
template<>
struct TLaneBySize_<sizeof(uint64_t)> {
    using UInt = uint64_t;
    using Int = int64_t;
    using FP = double;
};

template <typename T, size_t size, typename BySize_ = TLaneBySize_<size>>
using TLaneBySize = std::conditional_t<
    std::is_floating_point_v<T>, typename BySize_::FP,
    std::conditional_t<
        std::is_unsigned_v<T>, typename BySize_::UInt, typename BySize_::Int
    >
>;
} // namespace helper
/**
 * Type alias to make an unsigned version of the given lane type.
 *
 * @tparam TLane The input lane type.
 * @return The corresponding unsigned lane type.
 */
template <typename TLane>
using MakeUnsigned = helper::TLaneBySize<unsigned, sizeof(TLane)>;
/**
 * Type alias to make a signed version of the given lane type.
 *
 * @tparam TLane The input lane type.
 * @return The corresponding signed lane type.
 */
template <typename TLane>
using MakeSigned = helper::TLaneBySize<signed, sizeof(TLane)>;
/**
 * Type alias to make a floating-point version of the given lane type.
 *
 * @tparam TLane The input lane type.
 * @return The corresponding floating-point lane type.
 */
template <typename TLane>
using MakeFloat = helper::TLaneBySize<float, sizeof(TLane)>;
/**
 * Type alias to double the size of the given lane type.
 *
 * @tparam TLane The input lane type.
 * @return The resulting lane type with double the size.
 */
template <typename TLane>
using DoubleIt = helper::TLaneBySize<TLane, sizeof(TLane) * 2>;
/**
 * Type alias to halve the size of the given lane type.
 *
 * @tparam TLane The input lane type.
 * @return The resulting lane type with half the size.
 */
template <typename TLane>
using HalveIt = helper::TLaneBySize<TLane, sizeof(TLane) / 2>;
/// @}

namespace helper {
template <typename TLane>
struct ExtensionInfo {
    static constexpr bool kSupport = false;
};
template<>
struct ExtensionInfo<uint8_t> {
    using Vec = npyv_u8;
    using Vec2 = npyv_u8x2;
    using Vec3 = npyv_u8x3;
    using Mask = npyv_b8;
    static constexpr bool kSupport = true;
    static constexpr size_t kMaxLanes = npyv_nlanes_u8;
};
template<>
struct ExtensionInfo<int8_t> {
    using Vec = npyv_s8;
    using Vec2 = npyv_s8x2;
    using Vec3 = npyv_s8x3;
    static constexpr bool kSupport = true;
    static constexpr size_t kMaxLanes = npyv_nlanes_s8;
};
template<>
struct ExtensionInfo<uint16_t> {
    using Vec = npyv_u16;
    using Vec2 = npyv_u16x2;
    using Vec3 = npyv_u16x3;
    using Mask = npyv_b16;
    static constexpr bool kSupport = true;
    static constexpr size_t kMaxLanes = npyv_nlanes_u16;
};
template<>
struct ExtensionInfo<int16_t> {
    using Vec = npyv_s16;
    using Vec2 = npyv_s16x2;
    using Vec3 = npyv_s16x3;
    static constexpr bool kSupport = true;
    static constexpr size_t kMaxLanes = npyv_nlanes_s16;
};
template<>
struct ExtensionInfo<uint32_t> {
    using Vec = npyv_u32;
    using Vec2 = npyv_u32x2;
    using Vec3 = npyv_u32x3;
    using Mask = npyv_b32;
    static constexpr bool kSupport = true;
    static constexpr size_t kMaxLanes = npyv_nlanes_u32;
};
template<>
struct ExtensionInfo<int32_t> {
    using Vec = npyv_s32;
    using Vec2 = npyv_s32x2;
    using Vec3 = npyv_s32x3;
    static constexpr bool kSupport = true;
    static constexpr size_t kMaxLanes = npyv_nlanes_s32;
};
template<>
struct ExtensionInfo<uint64_t> {
    using Vec = npyv_u64;
    using Vec2 = npyv_u64x2;
    using Vec3 = npyv_u64x3;
    using Mask = npyv_b64;
    static constexpr bool kSupport = true;
    static constexpr size_t kMaxLanes = npyv_nlanes_u64;
};
template<>
struct ExtensionInfo<int64_t> {
    using Vec = npyv_s64;
    using Vec2 = npyv_s64x2;
    using Vec3 = npyv_s64x3;
    static constexpr bool kSupport = true;
    static constexpr size_t kMaxLanes = npyv_nlanes_s64;
};

#if NPY_SIMD_F32
template<>
struct ExtensionInfo<float> {
    using Vec = npyv_f32;
    using Vec2 = npyv_f32x2;
    using Vec3 = npyv_f32x3;
    static constexpr bool kSupport = true;
    static constexpr size_t kMaxLanes = npyv_nlanes_f32;
};
#endif
#if NPY_SIMD_F64
template<>
struct ExtensionInfo<double> {
    using Vec = npyv_f64;
    using Vec2 = npyv_f64x2;
    using Vec3 = npyv_f64x3;
    static constexpr bool kSupport = true;
    static constexpr size_t kMaxLanes = npyv_nlanes_f64;
};
#endif
} // namespace helper

/**@addtogroup cpp_simd_types Types
 * @ingroup cpp_simd
 *
 * Universal intrinsics introduce several data types to facilitate vectorized operations
 * and provide a uniform interface for performing vectorized computations across different platforms.
 * However, there are some usage considerations to keep in mind, particularly when dealing with
 * sizeless SIMD extensions:
 *
 * - These data types are not valid as members of any kind of structures or containers,
 *   such as std::tuple. They are designed to be used as standalone vectors or masks.
 *
 * - They cannot be used as dynamic or fixed arrays. The size of the data types may not known at compile time,
 *   so they cannot be used as the element type of an array.
 *
 * - The size of these data types shouldn't be determined using the `sizeof` operator. They are dynamically
 *   sized based on the enabled SIMD extension, and their size cannot be statically known.
 *
 * - They cannot be defined as static or thread-local storage variables. These data types rely on runtime
 *   information and are not suitable for static or thread-local storage.
 *
 * - It is not recommended to perform arithmetic operations directly on pointers to these data types.
 *   Pointer arithmetic may not produce the desired results due to the varying size and layout of the data types.
 *
 * It is important to consider these usage considerations when working with universal intrinsics
 * to ensure correct and efficient vectorized computations across different platforms and SIMD extensions.
 *
 * @{
 */
/**
 * Represents an N-lane vector based on the specified lane type.
 *
 * This template type represents a vector that consists of N lanes, where the value of N
 * depends on the enabled SIMD extension. For certain SIMD extensions such as SVE and RVV,
 * the number of lanes cannot be identified at compile-time. You can refer to `np::simd::NLanes`
 * and `np::simd::kMaxLanes` for more information.
 *
 * @note
 * This type may be defined as a template class or a template type alias depending on the enabled extensions.
 * Therefore, the template parameter `TLane` cannot be deduced by template specialization.
 * However, you can use `np::simd::GetLaneType` to determine the appropriate lane type.
 *
 * @note
 * This type does not provide any constructors or operators due to limitations with sizeless SIMD extensions.
 * It can be initialized using intrinsics such as `np::simd::Set`, `np::simd::Zero`, `np::simd::Undef`,
 * or through memory loaders such as `np::simd::Load`.
 *
 * @tparam TLane The lane type of the vector. Accepts fixed-size integer data types
 *               (`int8_t`, `int16_t`, `int32_t`, `int64_t`, `uint8_t`, `uint16_t`, `uint32_t`,
 *               `uint64_t`) as well as floating-point types (`float`, `double`).
 */
template <typename TLane>
class Vec final {
/// @cond
  public:
    using Reg = typename helper::ExtensionInfo<TLane>::Vec;
    Vec() = delete;
    static void *operator new     (size_t) = delete;
    static void *operator new[]   (size_t) = delete;
    static void  operator delete  (void*)  = delete;
    static void  operator delete[](void*)  = delete;
    NPY_FINLINE explicit Vec(Reg v) : val(v)
    {}
    Reg val;
/// @endcond
};
/**
 * Represents a tuple of two N-lane vectors.
 *
 * @note
 * This type may be defined as a template class or a template type alias depending on the enabled extensions.
 * Therefore, the template parameter `TLane` cannot be deduced by template specialization.
 * However, you can use `np::simd::GetLaneType` to determine the appropriate lane type.
 *
 * @note
 * This type does not provide any constructors or operators due to limitations with sizeless SIMD extensions.
 * It can be initialized using intrinsics such as `np::simd::SetTuple`, or through memory loaders
 * such as `np::simd::LoadDeinter2`, and N-lane vectors can be extracted using `np::simd::GetTuple`.
 *
 * @tparam TLane The lane type of the vector. Accepts fixed-size integer data types
 *               (`int8_t`, `int16_t`, `int32_t`, `int64_t`, `uint8_t`, `uint16_t`, `uint32_t`,
 *               `uint64_t`) as well as floating-point types (`float`, `double`).
 */
template <typename TLane>
class Vec2 final {
/// @cond
  public:
    Vec2() = delete;
    static void *operator new     (size_t) = delete;
    static void *operator new[]   (size_t) = delete;
    static void  operator delete  (void*)  = delete;
    static void  operator delete[](void*)  = delete;
    Vec<TLane> val[2];
/// @endcond
};
/**
 * Represents a tuple of three N-lane vectors.
 *
 * Similar to `Vec2` except for the number of N-lane vectors.
 */
template <typename TLane>
class Vec3 final {
/// @cond
 public:
    Vec3() = delete;
    static void *operator new     (size_t) = delete;
    static void *operator new[]   (size_t) = delete;
    static void  operator delete  (void*)  = delete;
    static void  operator delete[](void*)  = delete;
    Vec<TLane> val[3];
/// @endcond
};

/**
 * Represents a tuple of four N-lane vectors.
 *
 * Similar to `Vec2` except for the number of N-lane vectors.
 */
template <typename TLane>
class Vec4 final {
/// @cond
 public:
    Vec4() = delete;
    static void *operator new     (size_t) = delete;
    static void *operator new[]   (size_t) = delete;
    static void  operator delete  (void*)  = delete;
    static void  operator delete[](void*)  = delete;
    Vec<TLane> val[4];
/// @endcond
};

namespace helper {
template<typename TLane>
class Mask_ final {
 public:
    using Reg = typename ExtensionInfo<TLane>::Mask;
    Mask_() = delete;
    static void *operator new     (size_t) = delete;
    static void *operator new[]   (size_t) = delete;
    static void  operator delete  (void*)  = delete;
    static void  operator delete[](void*)  = delete;
    NPY_FINLINE explicit Mask_(Reg v) : val(v)
    {}
    Reg val;
};
} // namespace helper
/**
 * Represents a mask vector with boolean values or as a bitmask.
 *
 * The type `Mask` represents a mask vector where each lane or bit represents a condition or mask for
 * corresponding elements in other vectors.
 *
 * The `Mask` class uses the `TLane` template parameter, which accepts the same fixed-size integer and
 * floating-point types as the `Vec` class. However, the underlying representation of `TLane` may vary
 * depending on the enabled SIMD extension.
 *
 * For example, in some SIMD extensions, `TLane` is always converted to `uint8_t`,
 * while in others it may be converted to `uint8_t`, `uint16_t`, `uint32_t`, or
 * `uint64_t` based on the size of `TLane`.
 *
 * It is important to note that `Mask<TLane>` is not a strong data type and usually
 * requires an extra tag for any functionality that can't deduce its actual `TLane`.
 *
 * @note
 * template alias `np::simd::GetLaneType` returns the underlying lane type.
 *
 * @note
 * This type does not provide any constructors or operators due to limitations with sizeless SIMD extensions.
 * It can be initialized using `np::simd::ToMask` it can be also converted into `Vec<TLane>`
 * using `np::simd::ToVec`.
 *
 * @tparam TLane The lane type of the mask. Accepts fixed-size integer data types
 *               (`int8_t`, `int16_t`, `int32_t`, `int64_t`, `uint8_t`, `uint16_t`, `uint32_t`,
 *               `uint64_t`) as well as floating-point types (`float`, `double`).
 */
template<typename TLane>
#ifdef NPY_DOXYGEN
class Mask;
#elif NPY_SIMD_STRONG_MASK
using Mask = helper::Mask_<MakeUnsigned<TLane>>;
#else
using Mask = helper::Mask_<uint8_t>;
#endif

/// @}

/// @addtogroup cpp_simd_utils Utils
/// @{
namespace helper {
template <typename TVec>
struct GetLaneType_;
#define NPYV_IMPL_GET_LANE_TYPE(TLANE)                                   \
    template <> struct GetLaneType_<Vec<TLANE>>{ using Type = TLANE; };  \
    template <> struct GetLaneType_<Vec2<TLANE>>{ using Type = TLANE; }; \
    template <> struct GetLaneType_<Vec3<TLANE>>{ using Type = TLANE; }; \
    template <> struct GetLaneType_<Vec4<TLANE>>{ using Type = TLANE; };
NPYV_IMPL_GET_LANE_TYPE(uint8_t)
NPYV_IMPL_GET_LANE_TYPE(int8_t)
NPYV_IMPL_GET_LANE_TYPE(uint16_t)
NPYV_IMPL_GET_LANE_TYPE(int16_t)
NPYV_IMPL_GET_LANE_TYPE(uint32_t)
NPYV_IMPL_GET_LANE_TYPE(int32_t)
NPYV_IMPL_GET_LANE_TYPE(uint64_t)
NPYV_IMPL_GET_LANE_TYPE(int64_t)
#if NPY_SIMD_F32
    NPYV_IMPL_GET_LANE_TYPE(float)
#endif
#if NPY_SIMD_F64
    NPYV_IMPL_GET_LANE_TYPE(double)
#endif
template <> struct GetLaneType_<Mask<uint8_t>>  {using Type = uint8_t;};
#if NPY_SIMD_STRONG_MASK
template <> struct GetLaneType_<Mask<uint16_t>> {using Type = uint16_t;};
template <> struct GetLaneType_<Mask<uint32_t>> {using Type = uint32_t;};
template <> struct GetLaneType_<Mask<uint64_t>> {using Type = uint64_t;};
#endif
} // namespace helper
/// Type alias to get the lane type of all the supported Data types.
template <typename T>
using GetLaneType = typename helper::GetLaneType_<T>::Type;
/**
 * Determines whether the specified lane type is supported by the SIMD extension.
 *
 * @note
 * Most SIMD extensions support all types of lanes. However, there are some exceptions, such as NEON on armhf
 * that doesn't support single-precision operations, and VX (IBM Z) that doesn't support double-precision.
 *
 * @tparam TLane The lane type to check for support.
 */
template <typename TLane>
constexpr bool kSupportLane = helper::ExtensionInfo<TLane>::kSupport;
/**
 * The maximum number of lanes supported by the SIMD extension for the specified lane type.
 *
 * For non-sizeless SIMD extensions, it returns the same exact value as `np::simd::NLanes`.
 * However, for sizeless SIMD extensions, such as SVE, it returns the maximum possible number of
 * lanes that can be supported, which is calculated as `2048 / 8 / sizeof(TLane)`.
 *
 * @tparam TLane The lane type for which to determine the maximum number of lanes.
 */
template <typename TLane>
constexpr size_t kMaxLanes = helper::ExtensionInfo<TLane>::kMaxLanes;
/**
 * Indicates whether the enabled SIMD extension supports native Fused Multiply-Add (FMA) instructions.
 *
 * If the value is @c true, the extension natively supports FMA. If the value is @c false,
 * FMA is not supported natively, but it may be emulated using fast intrinsics.
 * However, it is important to note that the precision of the emulated FMA may not be the same as native support.
 */
constexpr bool kSupportFMA = NPY_SIMD_FMA3 != 0;
/**
 * Indicates whether the supported comparison intrinsics
 * (@c np::simd::Lt, @c np::simd::Le, @c np::simd::Gt, @c np::simd::Ge)
 * raise a floating-point invalid exception.
 *
 * For quiet NaNs. If the value is @c true, the comparison intrinsics will signal the invalid exception for quiet NaNs.
 * If the value is @c false, the exception will not be signaled.
 */
constexpr bool kCMPSignal = NPY_SIMD_CMPSIGNAL != 0;
/**
 * Indicates whether the enabled SIMD extension is running on big-endian mode.
 *
 * If the value is @c true, the extension is running on big-endian mode. If the value is @c false,
 * the extension is running on little-endian mode.
 */
constexpr bool kBigEndian = NPY_SIMD_BIGENDIAN != 0;
/**
 * Indicates whether the enabled SIMD extension supports strong-typed @c Mask.
 *
 * If the value is @c true, the extension supports strong-typed @c Mask. If the value is @c false,
 * the extension does not support strong-typed @c Mask.
 * Strong-typed @c Mask provides stricter type checking and can help prevent type-related errors.
 */
constexpr bool kStrongMask = NPY_SIMD_STRONG_MASK != 0;

/// @}
} // namespace npy::simd_ext

#endif // NPY_SIMD
#endif // NUMPY_CORE_SRC_COMMON_SIMD_WRAPPER_DATATYPES_HPP_
