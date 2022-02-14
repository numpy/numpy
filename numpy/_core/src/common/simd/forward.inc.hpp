#ifndef NUMPY_CORE_SRC_COMMON_SIMD_FORWARD_INC_HPP_
#define NUMPY_CORE_SRC_COMMON_SIMD_FORWARD_INC_HPP_

#if !NPY_SIMD
    #error "Not a standalone header, use simd/simd.hpp instead"
#endif

namespace np::NPY_CPU_DISPATCH_CURFX(simd_ext) {
/**@addtogroup cpp_simd_misc Miscellaneous
 * @ingroup cpp_simd
 * @rst
 * Index
 * -----
 * @endrst
 *
 * | Name                |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
 * |:--------------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
 * | @ref Width          |   |   |   |   |   |   |   |   |   |   |
 * | @ref NLanes         | x | x | x | x | x | x | x | x | x | x |
 * | @ref Undef          | x | x | x | x | x | x | x | x | x | x |
 * | @ref Zero           | x | x | x | x | x | x | x | x | x | x |
 * | @ref Set            | x | x | x | x | x | x | x | x | x | x |
 * | @ref Get0           | x | x | x | x | x | x | x | x | x | x |
 * | @ref SetTuple       | x | x | x | x | x | x | x | x | x | x |
 * | @ref GetTuple       | x | x | x | x | x | x | x | x | x | x |
 * | @ref Select         | x | x | x | x | x | x | x | x | x | x |
 * | @ref Reinterpret    | x | x | x | x | x | x | x | x | x | x |
 * | @ref Cleanup        |   |   |   |   |   |   |   |   |   |   |
 *
 * @rst
 * Definitions
 * -----------
 * @endrst
 *
 * @{
 */
/**
 * Returns the SIMD width in bytes.
 */
NPY_FINLINE size_t Width();
/**
 * Returns the number of vector lanes based on the lane type.
 *
 * @tparam TLane The lane type.
 * @param tag An optional tag to specify the lane type.
 * @return The number of vector lanes.
 *
 * @note Scheme:
 *       @code
 *       NLanes<float>(); // On AVX2, returns 8.
 *       @endcode
 */
template<typename TLane>
NPY_FINLINE size_t NLanes(TLane tag = 0);
/**
 * Returns an uninitialized N-lane vector.
 *
 * @tparam TLane The lane type.
 * @param tag An optional tag to specify the lane type.
 * @return The uninitialized N-lane vector.
 */
template<typename TLane>
NPY_FINLINE Vec<TLane> Undef(TLane tag = 0);
/**
 * Returns an N-lane vector with all lanes set to zero.
 *
 * @tparam TLane The lane type.
 * @param tag An optional tag to specify the lane type.
 * @return The N-lane vector with all lanes set to zero.
 *
 * @par Scheme:
 * @code
 * Zero<double>();
 * -------------------
 * Vec<double> {0.0, 0.0, ...}
 * @endcode
 */
template<typename TLane>
NPY_FINLINE Vec<TLane> Zero(TLane tag = 0);
/**
 * Initializes an N-lane vector with lanes set to specified values.
 *
 * Since the number of vector lanes is not fixed,
 * the behavior of this intrinsic is based on repeating
 * the remaining values based on the specified values.
 *
 * This intrinsic accepts a variable number of arguments:
 * 1, 2, 4, 8, 16, 32, or 64. The type of the vector will be
 * deduced based on the first argument, and the rest of the
 * arguments will be statically casted to match the type of
 * the first argument. Note that any arguments exceeding
 * the number of vector lanes will be ignored.
 *
 *
 * @tparam TLane0 The type of the first lane,
 *         and used to deduce the vector type.
 * @tparam TLane The types of the remaining lanes.
 * @param a The value for the first lane.
 * @param args The values for the remaining lanes.
 * @return The N-lane vector with lanes set to the specified values.
 *
 * @par Scheme:
 * @code
 * Set(1.0f);
 * -----------------------------
 * Vec<float> {1.0f, 1.0f, ...}
 * @endcode
 * @code
 * Set(uint32_t(1), 2);
 * -----------------------------
 * Vec<uint32_t> {1, 2, 1, 2, ...}
 * @endcode
 * @code
 * Set(1.0f, 2.0, 3.0, 4.0);
 * -----------------------------
 * Vec<float> {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, ...}
 * @endcode
 *
 */
template<typename TLane0, typename ...TLane>
NPY_FINLINE Vec<TLane0> Set(TLane0 a, TLane ...args)
{
    constexpr size_t narg = sizeof...(TLane) + 1;
    static_assert(
        narg == 1 || narg == 2 || narg == 4 || narg == 8 || narg == 16 ||
        narg == 32 || narg == 64,
        "Number of arguments should one of 2, 4, 8, 16, 32 or 64."
    );
    return Set(a, static_cast<TLane0>(args)...);
}
/**
 * Extracts the value of the first lane from an N-lane vector.
 *
 * @tparam TVec The type of the input vector.
 * @tparam TLane The lane type of the vector deduced from @tparam TVec.
 * @param a The input vector.
 * @return The value of the first lane from the input vector.
 */
template<typename TVec, typename TLane = GetLaneType<TVec>>
NPY_FINLINE TLane Get0(const TVec &a);
/**
 * Initializes a tuple of two N-lane vectors.
 *
 * This intrinsic creates a tuple of two N-lane vectors, where each vector
 * is initialized with the corresponding input values.
 *
 * @tparam TVec The type of the input vectors.
 * @tparam TLane The lane type of the vectors deduced from @tparam TVec.
 * @param a The first N-lane vector.
 * @param b The second N-lane vector.
 * @return Vec2<TLane> A tuple of two N-lane vectors.
 *
 * @par Scheme:
 * @code
 * SetTuple(Set(1.0), Set(2.0));
 * -------------------------
 * Vec2<double> {{1.0, 1.0, ...}, {2.0, 2.0, ...}}
 * @endcode
 */
template<typename TVec, typename TLane = GetLaneType<TVec>>
NPY_FINLINE Vec2<TLane> SetTuple(const TVec &a, const TVec &b);
/**
 * Initializes a tuple of three N-lane vectors.
 *
 * This intrinsic creates a tuple of three N-lane vectors, where each vector
 * is initialized with the corresponding input values.
 *
 * @tparam TVec The type of the input vectors.
 * @tparam TLane The lane type of the vectors deduced from @tparam TVec.
 * @param a The first N-lane vector.
 * @param b The second N-lane vector.
 * @param c The third N-lane vector.
 * @return Vec3<TLane> A tuple of three N-lane vectors.
 *
 * @par Scheme:
 * @code
 * SetTuple(Set(1.0), Set(2.0), Set(3.0));
 * -----------------------------------
 * Vec3<double> {{1.0, 1.0, ...}, {2.0, 2.0, ...}, {3.0, 3.0, ...}}
 * @endcode
 */
template<typename TVec, typename TLane = GetLaneType<TVec>>
NPY_FINLINE Vec3<TLane> SetTuple(const TVec &a, const TVec &b, const TVec &c);
/**
 * Initializes a tuple of four N-lane vectors.
 *
 * This intrinsic creates a tuple of three N-lane vectors, where each vector
 * is initialized with the corresponding input values.
 *
 * @tparam TVec The type of the input vectors.
 * @tparam TLane The lane type of the vectors deduced from @tparam TVec.
 * @param a The first N-lane vector.
 * @param b The second N-lane vector.
 * @param c The third N-lane vector.
 * @param d The fourth N-lane vector.
 * @return Vec4<TLane> A tuple of four N-lane vectors.
 *
 * @par Scheme:
 * @code
 * SetTuple(Set(1.0), Set(2.0), Set(3.0), Set(4.0));
 * -----------------------------------
 * Vec4<double> {{1.0, ...}, {2.0, ...}, {3.0, ...}, {4.0, ...}}
 * @endcode
 */
template<typename TVec, typename TLane = GetLaneType<TVec>>
NPY_FINLINE Vec4<TLane> SetTuple(const TVec &a, const TVec &b,
                                 const TVec &c, const TVec &d);

/**
 * Extracts an N-lane vector from a tuple of N-lane vectors.
 *
 * This intrinsic extracts the Nth vector from a tuple of N-lane vectors.
 *
 * @tparam Ind The index of the vector to extract.
 * @tparam TupleVec The type of the tuple of vectors.
 * @tparam TLane The lane type of the vectors deduced from @tparam TupleVec.
 * @param a The tuple of vectors.
 * @return Vec<TLane> The extracted N-lane vector.
 *
 *
 * @par Scheme:
 * @code
 * Vec3<double> tuple = SetTuple(Set(1.0), Set(2.0), Set(3.0));
 * GetTuple<0>(tuple);
 * -----------------------------------
 * Vec<double> {1.0, 1.0, ...}
 * @endcode
 */
template<int Ind, typename TupleVec, typename TLane = GetLaneType<TupleVec>>
NPY_FINLINE Vec<TLane> GetTuple(const TupleVec &a);
/**
 * Returns an N-lane vector set to the value of
 * `a` or `b` depending on the value of mask `m`.
 *
 * @tparam TVec The type of the input vectors.
 * @tparam TLane The lane type of the vector deduced from `TVec`.
 * @param m The mask indicating which lanes to select from `a` and `b`.
 * @param a The first N-lane vector.
 * @param b The second N-lane vector.
 *
 * @note Scheme:
 *       @code
 *       Vec<float> a = Set(1.0f, 2.0f);
 *       Vec<float> b = Set(2.0f, 1.0f);
 *       Select(Gt(a, b), a, b);
 *       ----------------------------------------
 *       Vec<float> {2.0f, 2.0f, 2.0f, 2.0f, ...}
 *       @endcode
 */
template<typename TVec, typename TLane = GetLaneType<TVec>>
NPY_FINLINE Vec<TLane> Select(const Mask<TLane> &m, const TVec &a,
                              const TVec &b);
/**
 * Convert an N-lane vector to a different type without
 * modifying the underlying data.
 *
 * @tparam TLane The type to reinterpret the vector lanes as.
 * @tparam TVec The type of the input vector.
 * @param a The input vector to reinterpret.
 * @param tag An optional tag parameter used for type deduction.
 * @return The N-lane vector with lanes reinterpreted as the specified type.
 *
 * @note Scheme:
 *       @code
 *       Reinterpret<uint32_t>(Set(-0.0f));
 *       ---------------------------------
 *       Vec<uint32_t> {0x80000000, 0x80000000, 0x80000000, 0x80000000, ...}
 *       @endcode
 */
template<typename TLane, typename TVec>
NPY_FINLINE Vec<TLane> Reinterpret(const TVec &a, TLane tag = 0);
/***************************
 * Extra
 ***************************/
/**
 * Zero the contents of all XMM or YMM registers,
 * to avoid the AVX-SSE transition penalty.
 *
 * @note this intrinsic should only be called one
 * time at the end of the SIMD kernel.
 */
 NPY_FINLINE void Cleanup();
/// @}

/**@addtogroup cpp_simd_memory Memory
 * @ingroup cpp_simd
 * @rst
 * Index
 * -----
 * @endrst
 *
 * | Name                  |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
 * |:----------------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
 * | @ref Load             | x | x | x | x | x | x | x | x | x | x |
 * | @ref LoadAligned      | x | x | x | x | x | x | x | x | x | x |
 * | @ref LoadStream       | x | x | x | x | x | x | x | x | x | x |
 * | @ref LoadLow          | x | x | x | x | x | x | x | x | x | x |
 * | @ref LoadDeinter2     | x | x | x | x | x | x | x | x | x | x |
 * | @ref LoadTill         | - | - | - | - | x | x | x | x | x | x |
 * | @ref LoadPairTill     | - | - | - | - | x | x | x | x | x | x |
 * | @ref IsLoadable       | - | - | - | - | x | x | x | x | x | x |
 * | @ref Loadn            | - | - | - | - | x | x | x | x | x | x |
 * | @ref LoadnTill        | - | - | - | - | x | x | x | x | x | x |
 * | @ref LoadnPair        | - | - | - | - | x | x | x | x | x | x |
 * | @ref LoadnPairTill    | - | - | - | - | x | x | x | x | x | x |
 * | @ref Store            | x | x | x | x | x | x | x | x | x | x |
 * | @ref StoreStream      | x | x | x | x | x | x | x | x | x | x |
 * | @ref StoreAligned     | x | x | x | x | x | x | x | x | x | x |
 * | @ref StoreLow         | x | x | x | x | x | x | x | x | x | x |
 * | @ref StoreHigh        | x | x | x | x | x | x | x | x | x | x |
 * | @ref StoreInter2      | x | x | x | x | x | x | x | x | x | x |
 * | @ref StoreTill        | - | - | - | - | x | x | x | x | x | x |
 * | @ref StorePairTill    | - | - | - | - | x | x | x | x | x | x |
 * | @ref IsStorable       | - | - | - | - | x | x | x | x | x | x |
 * | @ref Storen           | - | - | - | - | x | x | x | x | x | x |
 * | @ref StorenTill       | - | - | - | - | x | x | x | x | x | x |
 * | @ref StorenPair       | - | - | - | - | x | x | x | x | x | x |
 * | @ref StorenPairTill   | - | - | - | - | x | x | x | x | x | x |
 * | @ref Lookup128        | - | - | - | - | x | x | x | x | x | x |
 *
 *
 * @rst
 * Definitions
 * -----------
 * @endrst
 *
 * @{
 */
/***************************
 * Contiguous
 ***************************/
/**
 * Load the contents of an N-lane vector from unaligned memory.
 *
 * @param ptr Pointer to the memory block containing the data.
 * @return N-lane vector based on the type of the pointer.
 *
 * @note The load operation reads N consecutive elements from memory
 * starting at the specified pointer and constructs an N-lane vector.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:NLanes] = ptr[0:NLanes]
 * @endcode
 *
 * @par Example:
 * @code
 * const double *src;
 * const size_t nlanes = NLanes<double>();
 *
 * for (size_t i = 0, len = array_length & -nlanes; i < len; i += nlanes) {
 *     Vec<double> a = Load(src + i);
 *     // ...
 * }
 * @endcode
 */
template<typename TLane>
NPY_FINLINE Vec<TLane> Load(const TLane *ptr);
/**
 * Load N-lane vector contents from aligned memory.
 *
 * This intrinsic is similar to `np::simd::Load`, but the source
 * memory block must be aligned to `np::simd::Width()`.
 *
 * @tparam TLane The lane type of the vector and the type of the pointer.
 * @param ptr Pointer to the memory block containing the data.
 * @return N-lane vector based on the type of the pointer.
 *
 * @par Example:
 * @code
 * const double *src = ...;
 * Vec<double> a = LoadAligned(src);
 * // ...
 * @endcode
 */
template<typename TLane>
NPY_FINLINE Vec<TLane> LoadAligned(const TLane *ptr);
/**
 * Load N-lane vector contents from aligned memory
 * using a non-temporal memory hint.
 *
 * This intrinsic is similar to `np::simd::Load`,
 * but the source memory block must be aligned to `np::simd::Width()`.
 *
 * @tparam TLane The lane type of the vector and the type of the pointer.
 * @param ptr Pointer to the memory block containing the data.
 * @return N-lane vector based on the type of the pointer.
 *
 * @par Example:
 * @code{.cpp}
 * const double *src = ...;
 * Vec<double> a = LoadStream(src);
 * // ...
 * @endcode
 */
template<typename TLane>
NPY_FINLINE Vec<TLane> LoadStream(const TLane *ptr);
/**
 * Load half of N-lane vector contents, leaving the higher part undefined.
 *
 * This intrinsic loads the lower half of the N-lane vector contents from the
 * specified memory block, leaving the higher part of the vector undefined.
 *
 * @tparam TLane The lane type of the vector and the type of the pointer.
 * @param ptr Pointer to the memory block containing the data.
 * @return N-lane vector based on the type of the pointer.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:NLanes/2] = ptr[0:NLanes/2]
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * const int32_t *src;
 * const size_t nlanes = NLanes<int32_t>() / 2;
 *
 * for (size_t i = 0, len = array_length & -nlanes;
 *         i < len; i += nlanes) {
 *     Vec<int32_t> a = LoadLow(src + i);
 *     // ...
 * }
 * @endcode
 */
template<typename TLane>
NPY_FINLINE Vec<TLane> LoadLow(const TLane *ptr);
/**
 * Load and de-interleave a tuple of two N-lane vectors from memory.
 *
 * This intrinsic loads two interleaved N-lane vectors from memory
 * and de-interleaves them into a tuple of two separate N-lane vectors.
 *
 * @tparam TLane    The lane type of the vector.
 * @param ptr       Pointer to the memory block containing the interleaved vectors.
 * @return          A tuple of two N-lane vectors de-interleaved from the memory.
 *
 * @note This intrinsic only supports 32-bit and 64-bit data types.
 *
 * @par Example:
 * @code{.cpp}
 * int32_t src[] = {10, 100, 20, 200, 30, 300, 40, 400, ...};
 *
 * Vec2<int32_t> result = LoadDeinter2(src);
 * // result = {{10, 20, 30, 40, ...}, {100, 200, 300, 400, ...)}}
 * @endcode
 */
template<typename TLane>
NPY_FINLINE Vec2<TLane> LoadDeinter2(const TLane *ptr);
/**
 * Partially loads N-lane vector contents from memory.
 *
 * This intrinsic returns an N-lane vector with content loaded from the specified
 * memory block up to the number of lanes specified by the @p len parameter.
 *
 * If @p len is less than `np::simd::NLanes<TLane>()`, the remaining lanes of the returned
 * N-lane vector are set to the value specified by the @p fill parameter.
 *
 * If @p len is greater than or equal to `np::simd::NLanes<TLane>()`, the full lanes are
 * loaded from memory.
 *
 * @tparam TLane The lane type of the vector and the type of the pointer.
 * @param ptr Pointer to the memory block containing the data.
 * @param len Number of lanes to be loaded. Must be greater than zero.
 * @param fill Value of the remaining lanes when `len` is less than `np::simd::NLanes<TLane>()`.
 * @return N-lane vector based on the type of the pointer.
 *
 * @note This intrinsic only supports 32-bit and 64-bit data types.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:len] = ptr[0:len]
 * Vec<TLane>[len:NLanes] = fill
 * @endcode
 *
 * @par Example:
 * @code
 * int32_t data[] = {10, 20, 30, 40};
 * LoadTill(data, 2, 1);
 * // Result: {10, 20, 1, 1, 1, 1, 1, ...}
 * @endcode
 *
 * @par Example:
 * @code
 * const double *src;
 * const size_t nlanes = NLanes<double>();
 *
 * for (ptrdiff_t len = array_length; len > 0; len -= nlanes, src += nlanes) {
 *     Vec<double> a = LoadTill(src, len, 1.0);
 *     // ...
 * }
 * @endcode
 */
template<typename TLane>
NPY_FINLINE Vec<TLane> LoadTill(const TLane *ptr, size_t len, TLane fill);
/**
 * @overload
 * Partially loads N-lane vector contents from memory.
 *
 * This overload of the intrinsic is similar to the previous one,
 * except that the remaining lanes are set to zero when @p len is less than
 * `np::simd::NLanes<TLane>()`.
 *
 * @tparam TLane The lane type of the vector and the type of the pointer.
 * @param ptr Pointer to the memory block containing the data.
 * @param len Number of lanes to be loaded. Must be greater than zero.
 * @return N-lane vector based on the type of the pointer.
 *
 * @note This intrinsic only supports 32-bit and 64-bit data types.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:min(len, NLanes)] = ptr[0:min(len, NLanes)]
 * Vec<TLane>[min(len, NLanes):NLanes] = 0
 * @endcode
 *
 * @par Example:
 * @code
 * int32_t data[] = {10, 20, 30, 40};
 * LoadTill(data, 2);
 * // Result: {10, 20, 0, 0, 0, 0, 0, ...}
 * @endcode
 *
 * @code
 * const double *src;
 * const size_t nlanes = NLanes<double>();
 *
 * for (ptrdiff_t len = array_length; len > 0; len -= nlanes, src += nlanes) {
 *     Vec<double> a = LoadTill(src, len);
 *     // ...
 * }
 * @endcode
 */
template<typename TLane>
NPY_FINLINE Vec<TLane> LoadTill(const TLane *ptr, size_t len);
/**
 * Partially pair loads N-lane vector contents from memory.
 *
 * This intrinsic performs a partial pair load operation, where it loads pairs of
 * N-lane vector lanes from the specified memory block up to the number of pairs
 * specified by the parameter @p plen.
 *
 * If @p plen is less than `np::simd::NLanes<TLane>() / 2`, then the remaining
 * pairs of the returned N-lane vector are set to the values specified by
 * the parameters @p fill0 and @p fill1.
 *
 * If @p plen is greater than or equal to `np::simd::NLanes<TLane>() / 2`,
 * the full pairs are loaded from memory.
 *
 * @tparam TLane The lane type of the vector.
 * @param ptr    Pointer to the memory block containing the data.
 * @param plen   Number of pairs to be loaded. Must be greater than zero.
 * @param fill0  Value of the remaining even-indexed lanes when plen is less than
 *               `np::simd::Nlanes<TLane>() / 2`.
 * @param fill1  Value of the remaining odd-indexed lanes when plen is less than
 *               `np::simd::Nlanes<TLane>() / 2`.
 * @return N-lane vector based on the type of the pointer.
 *
 * @note This intrinsic only supports 32-bit and 64-bit data types.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:min(plen*2, NLanes)] = ptr[0:min(plen*2, NLanes)]
 * Vec<TLane>[min(plen*2, NLanes):NLanes] = {fill0, fill1}
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * int32_t data[] = {10, 20, 30, 40, 50, 60};
 * // plen = 1, fill0 = 1, fill1 = 2
 * LoadPairTill(data, 1, 1, 2);
 * // Result: {10, 20, 1, 2, 1, 2, ...}
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * const float *complex_array = ...;
 * const size_t half_nlanes = NLanes<float>() / 2;
 *
 * for (ptrdiff_t len = complex_array_length; len > 0;
 *               len -= half_nlanes, complex_array += half_nlanes) {
 *     Vec<float> a = LoadPairTill(complex_array, len, 1.0f, 2.0f);
 *     // ...
 * }
 * @endcode
 */
template<typename TLane>
NPY_FINLINE Vec<TLane> LoadPairTill(const TLane *ptr, size_t plen,
                                    TLane fill0, TLane fill1);
/**
 * @overload
 * Partially pair loads N-lane vector contents from memory.
 *
 * This overload of the intrinsic performs a partial pair load operation, where it
 * loads pairs of N-lane vector lanes from the specified memory block up to the
 * number of pairs specified by the parameter @p plen.
 *
 * If @p plen is less than `np::simd::NLanes<TLane>() / 2`, then the remaining
 * pairs of the returned N-lane vector are set to zero.
 *
 * If @p plen is greater than or equal to `np::simd::NLanes<TLane>() / 2`,
 * the full pairs are loaded from memory.
 *
 * @tparam TLane The lane type of the vector.
 * @param ptr      Pointer to the memory block containing the data.
 * @param plen Number of pairs to be loaded. Must be greater than zero.
 * @return N-lane vector based on the type of the pointer.
 *
 * @note This intrinsic only supports 32-bit and 64-bit data types.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:min(plen*2, NLanes)] = ptr[0:min(plen*2, NLanes)]
 * Vec<TLane>[min(plen*2, NLanes):NLanes] = 0
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * int32_t data[] = {10, 20, 30, 40, 50, 60};
 *
 * // Example 1: plen = 1
 * LoadPairTill(data, 1);
 * // Result: {10, 20, 0, 0, 0, 0, ...}
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * const float *complex_array = ...;
 * const size_t half_nlanes = NLanes<float>() / 2;
 *
 * for (ptrdiff_t len = complex_array_length; len > 0;
 *               len -= half_nlanes, complex_array += half_nlanes) {
 *     Vec<float> a = LoadPairTill(complex_array, len);
 *     // ...
 * }
 * @endcode
 */
template<typename TLane>
NPY_FINLINE Vec<TLane> LoadPairTill(const TLane *ptr, size_t plen);
/**
 * Validate the stride of an array for non-continuous load intrinsics.
 *
 * This function checks whether the given stride is suitable for non-continuous
 * load operations with the specified lane type (@p TLane).
 *
 * @param stride The step size or stride of the array, based on the @p TLane
 *               type.
 * @param tag An optional tag value that can be used for additional
 *            customization or specialization.
 * @return @c true if the stride is compatible with non-continuous load
 *         intrinsics, @c false otherwise.
 *
 * @note The @p TLane type represents the lane data type, and it is used to
 *       determine the expected stride size based on the SIMD architecture and
 *       data alignment requirements.
 *
 * @par Example:
 * @code{.cpp}
 * // Check if a stride of 4 is loadable for a 32-bit integer lane type
 * bool loadable = IsLoadable<int32_t>(4);
 * if (loadable) {
 *     // Use non-continuous load intrinsics with stride 4
 *     // ...
 * } else {
 *     // Handle the case where the stride is not loadable
 *     // ...
 * }
 * @endcode
 */
template<typename TLane>
NPY_FINLINE bool IsLoadable(intptr_t stride, TLane tag = 0);
/**
 * Load N-lane vector contents from non-contiguous memory block.
 *
 * This intrinsic loads an N-lane vector from a memory block where the lanes
 * are not stored contiguously. The loading is performed with a specified stride,
 * which represents the step size based on the pointer data type.
 *
 * @tparam TLane The lane type of the vector.
 * @param ptr    Pointer to the memory block with data.
 * @param stride The stride, which is the step size based on the pointer data type.
 * @return N-lane vector based on the type of the pointer.
 *
 * @note The stride value must be validated using `np::simd::IsLoadable<TLane>()`.
 * @note This intrinsic only supports 32-bit and 64-bit data types.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:NLanes] = ptr[0:stride*NLanes:stride]
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * int32_t data[] = {10, 20, 30, 40, 50, 60, ...};
 * Loadn(data, 2);
 * // Result: {10, 30, 50, ...}
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * const double *src;
 * intptr_t src_stride;
 *
 * const size_t nlanes = NLanes<double>();
 * const intptr_t step = nlanes * src_stride;
 *
 * if (IsLoadable<double>(src_stride)) {
 *     for (size_t len = array_length; len >= nlanes;
 *                 len -= nlanes, src += step) {
 *         Vec<double> a = Loadn(src, src_stride);
 *         // ...
 *     }
 * }
 * @endcode
 */
template<typename TLane>
NPY_FINLINE Vec<TLane> Loadn(const TLane *ptr, intptr_t stride);
/**
 * Partially loads N-lane vector contents from a non-contiguous memory block.
 *
 * This intrinsic is similar to `np::simd::LoadTill`, but it handles non-contiguous
 * memory blocks. It loads a portion of the N-lane vector from the specified
 * memory block with a specified stride. The number of lanes to be loaded is
 * determined by the parameter @p len.
 *
 * If @p len is less than the total number of lanes `np::simd::NLanes<TLane>()`,
 * the remaining lanes of the returned N-lane vector are set to the value specified
 * by the parameter @p fill.
 *
 * If @p len is greater than or equal to `np::simd::NLanes<TLane>()`, the full lanes are
 * loaded from the non-contiguous memory block according to the specified stride.
 *
 * @tparam TLane The lane type of the vector.
 * @param ptr Pointer to the memory block with data.
 * @param stride The stride, which is the step size based on the pointer data type.
 * @param len Length to be loaded. Must be greater than zero.
 * @param fill Value of the remaining lanes when @p len is less than `np::simd::NLanes<TLane>()`.
 * @return N-lane vector based on the type of the pointer.
 *
 * @note The stride value must be validated using `np::simd::IsLoadable<TLane>()`.
 * @note This intrinsic only supports 32-bit and 64-bit data types.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:min(len, NLanes)] = ptr[0:stride*min(len, NLanes):stride]
 * Vec<TLane>[min(len, NLanes):NLanes] = fill
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * int32_t data[] = {10, 20, 30, 40, 50, 60, ...};
 * LoadnTill(data, 2, 2, 1);
 * // Result: {10, 30, 1, 1, 1, 1, 1, 1, ...}
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * const double *src;
 * intptr_t src_stride;
 *
 * using vec = Vec<double>;
 * const size_t nlanes = vec::NLanes();
 * const intptr_t step = nlanes * src_stride;
 *
 * if (IsLoadable<double>(src_stride)) {
 *     for (ptrdiff_t len = array_length; len > 0;
 *                   len -= NLanes, src += step) {
 *         vec a = LoadnTill(src, src_stride, len, 1.0);
 *         // ...
 *     }
 * }
 * @endcode
 */
template<typename TLane>
NPY_FINLINE Vec<TLane> LoadnTill(const TLane *ptr, intptr_t stride, size_t len, TLane fill);
/**
 * @overload
 * Partially loads N-lane vector contents from non-contiguous memory block.
 *
 * This overload of the intrinsic is similar to the previous one,
 * except that the remaining lanes are set to zero when @p len is less than
 * `np::simd::NLanes<TLane>()`.
 *
 * @tparam TLane The lane type of the vector.
 * @param ptr    Pointer to the memory block with data.
 * @param stride The stride, which is the step size based on the pointer data type.
 * @param len    Length to be loaded. Must be greater than zero.
 * @return N-lane vector based on the type of the pointer.
 *
 * @note The stride value must be validated using `np::simd::IsLoadable<TLane>()`.
 * @note This intrinsic only supports 32-bit and 64-bit data types.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:min(len, NLanes)] = ptr[0:stride*min(len, NLanes):stride]
 * Vec<TLane>[min(len, NLanes):NLanes] = 0
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * int32_t data[] = {10, 20, 30, 40, 50, 60, ...};
 * LoadnTill(data, 2, 2);
 * // Result: {10, 30, 0, 0, 0, 0, 0, 0, ...}
 * @endcode
 */
template<typename TLane>
NPY_FINLINE Vec<TLane> LoadnTill(const TLane *ptr, intptr_t stride, size_t len);
/**
 * Pair loads N-lane vector contents from non-contiguous memory block.
 *
 * This intrinsic loads an N-lane vector from a memory block where the lanes
 * are not stored contiguously. The loading is performed with a specified stride,
 * which represents the step size of each two elements based on the pointer data type.
 *
 * @tparam TLane The lane type of the vector.
 * @param ptr    Pointer to the memory block with data.
 * @param stride The pair stride, which is the step size based on the pointer data type.
 * @return N-lane vector based on the type of the pointer.
 *
 * @note The @p stride value must be validated using `np::simd::IsLoadable<TLane>()`.
 * @note This intrinsic only supports 32-bit and 64-bit data types.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:Nlanes:2] = ptr[0:stride*Nlanes/2:stride]
 * Vec<TLane>[1:Nlanes:2] = ptr[1:stride*Nlanes/2:stride]
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * int32_t data[] = {10, 20, 30, 40, 50, 60, ...};
 *
 * LoadnPair(data, 1);
 * // Result: {10, 20, 20, 30, ...}
 *
 * LoadnPair(data, 2);
 * // Result: {10, 20, 30, 40, ...}
 *
 * LoadnPair(data, 3);
 * // Result: {10, 20, 40, 50, ...}
 * @endcode
 */
template<typename TLane>
NPY_FINLINE Vec<TLane> LoadnPair(const TLane *ptr, intptr_t stride);
/**
 * Partially pair loads N-lane vector contents from non-contiguous memory block.
 *
 * This intrinsic partially loads an N-lane vector from a memory block where the lanes
 * are not stored contiguously. The loading is performed with a specified pair stride,
 * which represents the step size of each two elements based on the pointer data type.
 * The number of pairs to be loaded is determined by the @p plen parameter.
 *
 * If @p plen is less than `np::simd::Nlanes<TLane>() / 2`, the remaining lanes of
 * the returned N-lane vector are set to the values specified by the @p fill0 and @p fill1.
 *
 * If @p len is greater than or equal to `np::simd::NLanes<TLane>() / 2`, the full lanes are
 * loaded from the non-contiguous memory block according to the specified stride.
 *
 * @tparam TLane  The lane type of the vector.
 * @param ptr     Pointer to the memory block with data.
 * @param stride  The pair stride, which is the step size based on the pointer data type.
 * @param plen    Number of pairs to be loaded. Must be greater than zero.
 * @param fill0   Value of the remaining even-indexed lanes when plen is less than
 *                np::simd::Nlanes<TLane>() / 2.
 * @param fill1   Value of the remaining odd-indexed lanes when plen is less than
 *                np::simd::Nlanes<TLane>() / 2.
 * @return N-lane vector based on the type of the pointer.
 *
 * @note The @p stride value must be validated using `np::simd::IsLoadable<TLane>()`.
 * @note This intrinsic only supports 32-bit and 64-bit data types.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:min(plen,NLanes/2)*2:2] = ptr[0:stride*min(plen,NLanes/2):stride]
 * Vec<TLane>[1:min(plen,NLanes/2)*2:2] = ptr[1:stride*min(plen,NLanes/2):stride]
 * Vec<TLane>[min(plen,NLanes/2)*2:NLanes] = {fill0, fill1, fill0, fill1, ...}
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * int32_t data[] = {10, 20, 30, 40, 50, 60, ...};
 *
 * LoadnPair(data, 1, 1, 1, 2);
 * // Result: {10, 20, 1, 2, ...}
 *
 * LoadnPair(data, 2, 3, 1, 2);
 * // Result: {10, 20, 30, 40, 50, 60, 1, 2, ...}
 *
 * LoadnPair(data, 3, 3, 1, 2);
 * // Result: {10, 20, 40, 50, 70, 80, 1, 2, ...}
 * @endcode
 */
template<typename TLane>
NPY_FINLINE Vec<TLane> LoadnPairTill(const TLane *ptr, intptr_t stride,
                                     size_t plen, TLane fill0, TLane fill1);
/**
 * @overload
 *
 * Partially pair loads N-lane vector contents from non-contiguous memory block.
 *
 * This overload is similar to the previous one, except the remaining lanes of
 * the returned N-lane vector are set to zero.
 *
 * @tparam TLane  The lane type of the vector.
 * @param ptr     Pointer to the memory block with data.
 * @param stride  The pair stride, which is the step size based on the pointer data type.
 * @param plen    Number of pairs to be loaded. Must be greater than zero.
 * @return N-lane vector based on the type of the pointer.
 *
 * @note The @p stride value must be validated using `np::simd::IsLoadable<TLane>()`.
 * @note This intrinsic only supports 32-bit and 64-bit data types.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:min(plen,NLanes/2)*2:2] = ptr[0:stride*min(plen,NLanes/2):stride]
 * Vec<TLane>[1:min(plen,NLanes/2)*2:2] = ptr[1:stride*min(plen,NLanes/2):stride]
 * Vec<TLane>[min(plen,NLanes/2)*2:NLanes] = {0, 0, 0, 0, ...}
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * int32_t data[] = {10, 20, 30, 40, 50, 60, ...};
 *
 * LoadnPair(data, 1, 1);
 * // Result: {10, 20, 0, 0, ...}
 *
 * LoadnPair(data, 2, 3);
 * // Result: {10, 20, 30, 40, 50, 60, 0, 0, ...}
 *
 * LoadnPair(data, 3, 3);
 * // Result: {10, 20, 40, 50, 70, 80, 0, 0, ...}
 * @endcode
 */
template<typename TLane>
NPY_FINLINE Vec<TLane> LoadnPairTill(const TLane *ptr, intptr_t stride, size_t plen);
/**
 * Store N-lane vector contents to memory.
 *
 * This intrinsic stores the contents of an N-lane vector @p a
 * to a memory block specified by @p ptr.
 *
 * @tparam TLane The lane type of the vector and the pointer.
 * @param ptr    Pointer to the memory block.
 * @param a      The N-lane vector.
 *
 * @par Scheme:
 * @code
 * ptr[0:NLanes] = a[0:NLanes]
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * const int16_t *src;
 * int16_t *dst;
 *
 * const size_t nlanes = NLanes();
 *
 * for (size_t i = 0, len = array_length & -nlanes; i < len; i += nlanes) {
 *     Vec<int16_t> a = Load(src + i);
 *     // ...
 *     Store(dst, a + i);
 * }
 * @endcode
 */
template<typename TLane>
NPY_FINLINE void Store(TLane *ptr, const Vec<TLane> &a);
/**
 * Store N-lane vector contents to aligned memory.
 *
 * This intrinsic stores the contents of an N-lane vector @p a to an aligned memory block specified by @p ptr.
 * The destination memory block should be aligned to the width of the vector (np::simd::Width()).
 *
 * @tparam TLane The lane type of the vector and the pointer.
 * @param ptr    Pointer to the aligned memory block.
 * @param a      The N-lane vector.
 *
 * @par Scheme:
 * @code
 * ptr[0:NLanes] = a[0:NLanes]
 * @endcode
 */
template<typename TLane>
NPY_FINLINE void StoreAligned(TLane *ptr, const Vec<TLane> &a);
/**
 * Store N-lane vector contents to aligned memory using a non-temporal memory hint.
 *
 * This intrinsic stores the contents of an N-lane vector @p a to an aligned memory block specified by @p ptr
 * using a non-temporal memory hint. The destination memory block should be aligned to the width of the vector (np::simd::Width()).
 *
 * @tparam TLane The lane type of the vector and the pointer.
 * @param ptr    Pointer to the aligned memory block.
 * @param a      The N-lane vector.
 *
 * @par Scheme:
 * @code
 * ptr[0:NLanes] = a[0:NLanes]
 * @endcode
 */
template<typename TLane>
NPY_FINLINE void StoreStream(TLane *ptr, const Vec<TLane> &a);
/**
 * Store the lower half of N-lane vector contents to memory.
 *
 * This intrinsic stores the lower half of an N-lane vector @p a to a memory block specified by @p ptr.
 * The lower half contains the first NLanes/2 elements of the vector.
 *
 * @tparam TLane The lane type of the vector and the pointer.
 * @param ptr    Pointer to the memory block.
 * @param a      The N-lane vector.
 *
 * @par Scheme:
 * @code
 * ptr[0:NLanes/2] = a[0:NLanes/2]
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * const int16_t *src;
 * int16_t *dst;
 *
 * const size_t half_nlanes = NLanes<TLane>() / 2;
 *
 * for (size_t i = 0, len = array_length & -half_nlanes; i < len; i += half_nlanes) {
 *     Vec<int16_t> a = Loadl(src + i);
 *     // ...
 *     StoreLow(dst, a + i);
 * }
 * @endcode
 */
template<typename TLane>
NPY_FINLINE void StoreLow(TLane *ptr, const Vec<TLane> &a);
/**
 * Store the higher half of N-lane vector contents to memory.
 *
 * This intrinsic stores the higher half of an N-lane vector @p a to a memory block specified by @p ptr.
 * The higher half contains the remaining NLanes/2 elements of the vector.
 *
 * @tparam TLane The lane type of the vector and the pointer.
 * @param ptr    Pointer to the memory block.
 * @param a      The N-lane vector.
 *
 * @par Scheme:
 * @code
 * ptr[0:NLanes/2] = a[NLanes/2:NLanes]
 * @endcode
 */
template<typename TLane>
NPY_FINLINE void StoreHigh(TLane *ptr, const Vec<TLane> &a);
/**
 * Interleave and store a tuple of two N-lane vector contents to memory.
 *
 * This intrinsic interleaves the lanes from two N-lane vectors `a`
 * and stores them as a tuple of two consecutive vectors in memory.
 * The resulting memory layout will have two interleaved vectors.
 *
 * @tparam TLane The lane type of the tuple of two N-lane vector and the pointer.
 * @param ptr    Pointer to the memory block where the data will be stored.
 * @param a      The tuple of two N-Lane vector to be interleaved and stored.
 *
 * @par Scheme:
 * @code
 * interleaved = interleave(GetTuple<0>(a), GetTuple<1>(a))
 * ptr[0:NLanes] = interleaved[0:NLanes]
 * ptr[NLanes:2*NLanes] = interleaved[NLanes:NLanes*2]
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * int32_t dst[MaxNLanes<TLane>];
 * Vec2<int32_t> a = SetTuple(
 *     Set(int32_t(10), 20, 30, 40, 50, 60, 70, 80, ...),
 *     Set(int32_t(100), 200, 300, 400, 500, 600, 700, 800, ...)
 * );
 *
 * StoreInter2(dst, a);
 * // Result: dst[] = {10, 100, 20, 200, 30, 300, 40, 400, 50, 500, 60, 600, 70, 700, 80, 800, ...}
 * @endcode
 */
template<typename TLane>
NPY_FINLINE void StoreInter2(TLane *ptr, const Vec2<TLane> &a);
/**
 * Partially store N-lane vector contents to memory.
 *
 * This intrinsic stores a partial N-lane vector @p a to a memory block specified by @p ptr.
 * The number of lanes to be stored is determined by the @p len parameter.
 *
 * If @p len is less than np::simd::NLanes<TLane>(),
 * the remaining lanes will be ignored, otherwise all lanes will be stored.
 *
 * If @p len is greater than or equal to `np::simd::NLanes<TLane>()`, the full lanes are
 * stored to the memory.
 *
 * @tparam TLane The lane type of the vector and the pointer.
 * @param ptr   Pointer to the memory block with data.
 * @param len   Number of lanes to be stored. Must be greater than zero.
 * @param a     The N-lane vector.
 *
 * @note This intrinsic only supports 32-bit and 64-bit data types.
 *
 * @par Scheme:
 * @code
 * ptr[0:min(len, NLanes)] = a[0:min(len, NLanes)]
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * int32_t dst[] = {1, 2, 3, 4, ...};
 * Vec<int32_t> a = Set(int32_t(10), 20, 30, 40, 50, 60, 70, 80, ...);
 * StoreTill(dst, 2)
 * // Result: {10, 20, 3, 4, ...}
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * const double *src, *dst;
 * const size_t nlanes = NLanes<double>();
 *
 * for (ptrdiff_t len = array_length; len > 0; len -= nlanes,
 *               src += nlanes, dst += nlanes) {
 *     Vec<double> a = LoadTill(src, len, 1.0);
 *     // Perform operations on 'a'
 *     StoreTill(dst, len, a);
 * }
 * @endcode
 */
template<typename TLane>
NPY_FINLINE void StoreTill(TLane *ptr, size_t len, const Vec<TLane> &a);
/**
 * Partially pair store N-lane vector contents to memory.
 *
 * This intrinsic partially stores an N-lane vector to a memory block, up to the specified number of pairs
 * specified by the @p plen parameter. The lanes are stored in pairs, where each pair is composed of
 * two consecutive lanes from the vector.
 *
 * If @p plen is less than `np::simd::Nlanes<TLane>() / 2`,
 * the remaining pairs of lanes in the vector are ignored,
 * otherwise all lanes will be stored.
 *
 * @tparam TLane  The lane type of the vector and the pointer.
 * @param ptr     Pointer to the memory block where the data will be stored.
 * @param plen    Number of pairs to be stored. Must be greater than zero.
 * @param a       The N-lane vector with the data to be stored.
 *
 * @note This intrinsic only supports 32-bit and 64-bit data types.
 *
 * @par Scheme:
 * @code
 * ptr[0:min(plen,NLanes/2)*2] = a[0:min(plen,NLanes/2)*2]
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * int32_t dst[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...};
 * Vec<int32_t> a = Set(int32_t(10), 20, 30, 40, 50, 60, 70, 80, 90, 100, ...);
 *
 * StorePairTill(dst, 2, a);
 * // Result: dst[] = {10, 20, 30, 40, 5, 6, 7, 8 ...}
 *
 * StorePairTill(dst, 3, a);
 * // Result: dst[] = {10, 20, 30, 40, 50, 60, 7, 8,  ...}
 *
 * StorePairTill(dst, 4, a);
 * // Result: dst[] = {10, 20, 30, 40, 50, 60, 70, 80, 9, 10, ...}
 * @endcode
 */
template<typename TLane>
NPY_FINLINE void StorePairTill(TLane *ptr, size_t plen, const Vec<TLane> &a);
/**
 * Validate the stride of an array for non-continuous store intrinsics.
 *
 * This function checks whether the given stride is suitable for non-continuous
 * store operations with the specified lane type (@p TLane).
 *
 * @param stride The step size or stride of the array, based on the @p TLane
 *               type.
 * @param tag An optional tag value that can be used for additional
 *            customization or specialization.
 * @return @c true if the stride is compatible with non-continuous store
 *         intrinsics, @c false otherwise.
 *
 * @note The @p TLane type represents the lane data type, and it is used to
 *       determine the expected stride size based on the SIMD architecture and
 *       data alignment requirements.
 *
 * @par Example:
 * @code{.cpp}
 * // Check if a stride of 4 is storable for a 32-bit integer lane type
 * bool storable = IsStorable<int32_t>(4);
 * if (storable) {
 *     // Use non-continuous store intrinsics with stride 4
 *     // ...
 * } else {
 *     // Handle the case where the stride is not storable
 *     // ...
 * }
 * @endcode
 */
template<typename TLane>
NPY_FINLINE bool IsStorable(intptr_t stride, TLane tag = 0);
/**
 * Store N-lane vector contents to non-contiguous memory.
 *
 * This intrinsic stores the contents of an N-lane vector to a memory
 * block with non-contiguous addresses. The stride parameter specifies the
 * step size between consecutive elements in the memory block.
 *
 * @tparam TLane    The lane type of the vector.
 * @param ptr       Pointer to the memory block where the data will be stored.
 * @param stride    The step size based on the pointer data type.
 * @param a         The N-lane vector with the data to be stored.
 *
 * @note The stride value needs to be validated using np::simd::IsStorable<TLane>().
 * @note This intrinsic only supports 32-bit and 64-bit data types.
 *
 * @par Scheme:
 * @code
 * ptr[0:stride*NLanes:stride] = a[0:NLanes]
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * const float *src;
 * float *dst;
 *
 * intptr_t src_stride, dst_stride;
 *
 * const size_t nlanes = NLanes<TLane>();
 * const intptr_t src_step = nlanes * src_stride;
 * const intptr_t dst_step = nlanes * dst_stride;
 *
 * if (IsLoadable<float>(src_stride) &&
 *     IsStorable<float>(dst_stride)
 * ) {
 *     for (size_t len = array_length; len >= nlanes;
 *             len -= nlanes, src += src_step,
 *             dst += dst_step
 *     ) {
 *         Vec<float> a = Loadn(src, src_stride);
 *         ...
 *         Storen(dst, dst_stride, a);
 *     }
 * }
 * @endcode
 */
template<typename TLane>
NPY_FINLINE void Storen(TLane *ptr, intptr_t stride, const Vec<TLane> &a);
/**
 * Partially store N-lane vector contents to non-contiguous memory.
 *
 * This intrinsic partially stores an N-lane vector to a memory block
 * with non-contiguous addresses, up to the specified number of lanes
 * specified by the len parameter. The stride parameter specifies the
 * step size between consecutive elements in the memory block.
 *
 * If @p len is less than `np::simd::Nlanes<TLane>()`,
 * the remaining pairs of lanes in the vector are ignored,
 * otherwise all lanes will be stored.
 *
 * @tparam TLane    The lane type of the vector and the pointer.
 * @param ptr       Pointer to the memory block where the data will be stored.
 * @param stride    The step size based on the pointer data type.
 * @param len       Number of lanes to be stored. Must be greater than zero.
 * @param a         The N-lane vector with the data to be stored.
 *
 * @note The stride value needs to be validated using np::simd::IsStorable<TLane>().
 * @note This intrinsic only supports 32-bit and 64-bit data types.
 *
 * @par Scheme:
 * @code
 * ptr[0:stride*min(len, NLanes):stride] = a[0:min(len, NLanes)]
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * int32_t dst[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...};
 * Vec<int32_t> a = Set(int32_t(10), 20, 30, 40, 50, 60, 70, 80, 90, 100, ...);
 *
 * StorenTill(dst, 2, 3, a);
 * // Result: dst[] = {10, 2, 20, 4, 30, 6, 7, 8, 9, 10, ...}
 *
 * StorenTill(dst, 3, 4, a);
 * // Result: dst[] = {10, 2, 3, 20, 5, 6, 30, 8, 9, 40, ...}
 * @endcode
 */
template<typename TLane>
NPY_FINLINE void StorenTill(TLane *ptr, intptr_t stride, size_t len, const Vec<TLane> &a);
/**
 * Pair store N-lane vector contents to non-contiguous memory.
 *
 * This intrinsic stores the N-lane vector contents to a memory block with non-contiguous addresses, using a
 * specified pair stride. The lanes are stored in pairs, where each pair is composed of two consecutive lanes
 * from the vector.
 *
 * @tparam TLane  The lane type of the vector and the pointer.
 * @param ptr     Pointer to the memory block where the data will be stored.
 * @param stride  The step size between consecutive pairs in the memory block.
 * @param a       The N-lane vector with the data to be stored.
 *
 * @note The stride value needs to be validated using np::simd::IsStorable<TLane>().
 * @note This intrinsic only supports 32-bit and 64-bit data types.
 *
 * @par Scheme:
 * @code
 * ptr[0:stride*NLanes/2:stride] = a[0:NLanes:2]
 * ptr[1:stride*NLanes/2:stride] = a[1:NLanes:2]
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * int32_t dst[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...};
 * Vec<int32_t> a = Set(int32_t(10), 20, 30, 40, 50, 60, 70, 80, 90, 100, ...);
 *
 * StorenPair(dst, 2, a);
 * // Result: dst[] = {10, 20, 30, 40, 50, 60, 70, 80, 9, 10, ...}
 *
 * StorenPair(dst, 3, a);
 * // Result: dst[] = {10, 20, 3, 30, 40, 6, 50, 60, 9, 70, 80, ...}
 * @endcode
 */
template<typename TLane>
NPY_FINLINE void StorenPair(TLane *ptr, intptr_t stride, const Vec<TLane> &a);
/**
 * Partially pair store N-lane vector contents to non-contiguous memory.
 *
 * This intrinsic partially stores an N-lane vector to a memory block
 * with non-contiguous addresses, using a specified pair stride.
 * The lanes are stored in pairs, where each pair is composed
 * of two consecutive lanes from the vector.
 *
 * If `plen` is less than `np::simd::NLanes<TLane>() / 2`,
 * the remaining pairs of lanes in the vector are ignored; otherwise,
 * all available pairs will be stored.
 *
 * @tparam TLane  The lane type of the vector and the pointer.
 * @param ptr     Pointer to the memory block where the data will be stored.
 * @param stride  The step size between consecutive pairs in the memory block.
 * @param plen    Number of pairs to be stored. Must be greater than zero.
 * @param a       The N-lane vector with the data to be stored.
 *
 * @note The stride value needs to be validated using np::simd::IsStorable<TLane>().
 * @note This intrinsic only supports 32-bit and 64-bit data types.
 *
 * @par Scheme:
 * @code
 * ptr[0:stride*min(plen,NLanes/2):stride] = a[0:min(plen,NLanes/2)*2:2]
 * ptr[1:stride*min(plen,NLanes/2):stride] = a[1:min(plen,NLanes/2)*2:2]
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * int32_t dst[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...};
 * Vec<int32_t> a = Set(int32_t(10), 20, 30, 40, 50, 60, 70, 80, 90, 100, ...);
 *
 * StorenPairTill(dst, 2, 2, a);
 * // Result: dst[] = {10, 20, 3, 30, 40, 6, 7, 8, 9, 10, ...}
 *
 * StorenPairTill(dst, 3, 4, a);
 * // Result: dst[] = {10, 20, 3, 30, 40, 6, 50, 60, 9, 70, 80, ...}
 * @endcode
 */
template<typename TLane>
NPY_FINLINE void StorenPairTill(TLane *ptr, intptr_t stride, size_t plen, const Vec<TLane> &a);
/**
 * Perform a lookup operation on a 128-byte table using lane indices.
 *
 * This intrinsic performs a lookup operation on a 128-byte table using the specified lane indices.
 * It returns a new vector where each lane contains the value from the table at the corresponding index.
 *
 * @tparam TLane  The lane type of the returned vector and the pointer.
 * @param table   Pointer to the table of values.
 * @param idx     A vector containing the unsigned integer lane indices.
 * @return        A new vector with the lookup results.
 *
 * @note This intrinsic only supports 32-bit and 64-bit data types.
 *
 * @par Scheme:
 * @code
 * Vec<TLane> v;
 * for (size_t i = 0; i < NLanes<TLane>(); ++i) {
 *     v[i] = table[idx[i]];
 * }
 * return v;
 * @endcode
 *
 * @par Example:
 * @code{.cpp}
 * float table[32] = {1, 2, 3, 4, 5, 6, 7, 8, ...};
 * Vec<uint32_t> indices = Set(uint32_t(2), 0, 3, 1, ...);
 * Vec<float> result = Lookup128(table, indices);
 * // Result: result = {3, 1, 4, 2, ...}
 * @endcode
 */
template<typename TLane>
NPY_FINLINE Vec<TLane> Lookup128(const TLane *table, const Vec<MakeUnsigned<TLane>> &idx);
/// @}

/**@addtogroup cpp_simd_bitwise Bitwise
 * @ingroup cpp_simd
 * @rst
 * Index
 * -----
 * @endrst
 *
 * | name                    |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
 * |-------------------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
 * | @ref And                | x | x | x | x | x | x | x | x | x | x |
 * | @ref Andc               | x | x | x | x | x | x | x | x | x | x |
 * | @ref Or                 | x | x | x | x | x | x | x | x | x | x |
 * | @ref Orc                | x | x | x | x | x | x | x | x | x | x |
 * | @ref Xor                | x | x | x | x | x | x | x | x | x | x |
 * | @ref Xnor               | x | x | x | x | x | x | x | x | x | x |
 * | @ref Not                | x | x | x | x | x | x | x | x | x | x |
 * | @ref Shr                | - | - | x | x | x | x | x | x | - | - |
 * | @ref Shl                | - | - | x | x | x | x | x | x | - | - |
 * | @ref Shli               | - | - | x | x | x | x | x | x | - | - |
 * | @ref Shri               | - | - | x | x | x | x | x | x | - | - |
 *
 * @rst
 * Definitions
 * -----------
 * @endrst
 * @{
 */
/***************************
 * Shifting
 ***************************/
/**
 * Bitwise shift left operation.
 *
 * This intrinsic performs a bitwise left shift operation on the given vector,
 * where each lane is shifted left by the specified number of bits.
 * This intrinsic supports 16-bit, 32-bit, and 64-bit integer datatypes.
 *
 * @tparam TVec  The N-lane vector type.
 * @param a      The input vector to be shifted.
 * @param n      The number of bits to shift left by.
 * @return       The resulting vector after the left shift operation.
 */
template<typename TVec>
NPY_FINLINE TVec Shl(const TVec &a, int n);
/**
 * Bitwise shift right operation.
 *
 * This intrinsic performs a bitwise right shift operation on the given vector,
 * where each lane is shifted right by the specified number of bits
 * This intrinsic supports 16-bit, 32-bit, and 64-bit integer datatypes.
 *
 * @note this intrinsic will perform arithmetic right shifting (sign-extension)
 * for signed integer datatypes.
 *
 * @tparam TVec  The N-lane vector type.
 * @param a      The input vector to be shifted.
 * @param n      The number of bits to shift right by.
 * @return       The resulting vector after the right shift operation.
 */
template<typename TVec>
NPY_FINLINE TVec Shr(const TVec &a, int n);
/**
 * Bitwise shift left operation by an immediate constant.
 *
 * This intrinsic performs a bitwise left shift operation on the given vector,
 * where each lane is shifted left by the specified immediate constant value.
 * This intrinsic supports 16-bit, 32-bit, and 64-bit integer datatypes.
 *
 * @tparam N    The immediate constant representing the number of bits to shift right by.
 *              It must be a non-zero postive integer and cannot exceed the width of the lane type.
 * @tparam TVec The N-lane vector type.
 * @param a     The input vector to be shifted.
 * @return      The resulting vector after the left shift operation.
 *
 * @note The value of N must be a non-zero positive integer and
 * cannot exceed the width of the lane type.
 */
template<int N, typename TVec>
NPY_FINLINE TVec Shli(const TVec &a);
/**
 * Bitwise shift right operation by an immediate constant.
 *
 * This intrinsic performs a bitwise right shift operation on the given vector,
 * where each lane is shifted right by the specified immediate constant value.
 * This intrinsic supports 16-bit, 32-bit, and 64-bit integer values.
 *
 * @tparam N    The immediate constant representing the number of bits to shift right by.
 *              It must be a non-zero postive integer and cannot exceed the width of the lane type.
 * @tparam TVec The N-lane vector type.
 * @param a     The input vector to be shifted.
 * @return      The resulting vector after the right shift operation.
 *
 * @note this intrinsic will perform arithmetic right shifting (sign-extension)
 * for signed integer datatypes.
 */
template<int N, typename TVec>
NPY_FINLINE TVec Shri(const TVec &a);
/***************************
 * Logical
 ***************************/
/**
 * Bitwise AND
 *
 * Performs a bitwise AND operation between two N-lane vectors or masks.
 *
 * @tparam TVecOrMask The N-lane vector or mask type.
 * @param a The first input N-lane vector or mask.
 * @param b The second input N-lane vector or mask.
 * @return The result of the bitwise AND operation.
 */
template<typename TVecOrMask>
NPY_FINLINE TVecOrMask And(const TVecOrMask &a, const TVecOrMask &b);

/**
 * Bitwise AND with Complement
 *
 * Performs a bitwise AND operation between the first input vector or mask, `a`,
 * and the complement of the second input vector or mask, `b`.
 *
 * @tparam TVecOrMask The N-lane vector or mask type.
 * @param a The first input N-lane vector or mask.
 * @param b The second input N-lane vector or mask.
 * @return The result of the bitwise AND with complement operation.
 */
template<typename TVecOrMask>
NPY_FINLINE TVecOrMask Andc(const TVecOrMask &a, const TVecOrMask &b);
/**
 * Bitwise OR
 *
 * Performs a bitwise OR operation between two N-lane vectors or masks.
 *
 * @tparam TVecOrMask The N-lane vector or mask type.
 * @param a The first input N-lane vector or mask.
 * @param b The second input N-lane vector or mask.
 * @return The result of the bitwise OR operation.
 */
template<typename TVecOrMask>
NPY_FINLINE TVecOrMask Or(const TVecOrMask &a, const TVecOrMask &b);

/**
 * Bitwise OR with Complement
 *
 * Performs a bitwise OR operation between the first input vector or mask, `a`,
 * and the complement of the second input vector or mask, `b`.
 *
 * @tparam TVecOrMask The N-lane vector or mask type.
 * @param a The first input N-lane vector or mask.
 * @param b The second input N-lane vector or mask.
 * @return The result of the bitwise OR with complement operation.
 */
template<typename TVecOrMask>
NPY_FINLINE TVecOrMask Orc(const TVecOrMask &a, const TVecOrMask &b);
/**
 * Bitwise XOR
 *
 * Performs a bitwise XOR operation between two N-lane vectors or masks.
 *
 * @tparam TVecOrMask The N-lane vector or mask type.
 * @param a The first input N-lane vector or mask.
 * @param b The second input N-lane vector or mask.
 * @return The result of the bitwise XOR operation.
 */
template<typename TVecOrMask>
NPY_FINLINE TVecOrMask Xor(const TVecOrMask &a, const TVecOrMask &b);
/**
 * @ingroup cpp_simd_bitwise
 * Bitwise XNOR
 *
 * Performs a bitwise XNOR (logical equivalence) operation between two N-lane vectors or masks.
 * It is equivalent to the negation of the bitwise XOR operation between the two inputs.
 *
 * @tparam TVecOrMask The N-lane vector or mask type.
 * @param a The first input N-lane vector or mask.
 * @param b The second input N-lane vector or mask.
 * @return The result of the bitwise XNOR operation.
 */
template<typename TVecOrMask>
NPY_FINLINE TVecOrMask Xnor(const TVecOrMask &a, const TVecOrMask &b);

/**
 * @ingroup cpp_simd_bitwise
 * Bitwise NOT
 *
 * Performs a bitwise NOT (complement) operation on a vector or mask.
 * It flips the bits of each lane in the input N-lane vector or mask.
 *
 * @tparam TVecOrMask The N-lane vector or mask type.
 * @param a The input N-lane vector or mask.
 * @return The result of the bitwise NOT operation.
 */
template<typename TVecOrMask>
NPY_FINLINE TVecOrMask Not(const TVecOrMask &a);
/// @}

/**@addtogroup cpp_simd_comparison Comparison
 * @ingroup cpp_simd
 *
 * The following intrinsics perform ordered comparisons for
 * floating-point data types. An ordered comparison considers
 * a result of @c false/0 if any of the corresponding lanes has @c NaN,
 * except for the @ref Ne operation which performs an unordered comparison.
 *
 * Regarding floating-point signaling, the behavior of
 * @ref Lt, @ref Le, @ref Gt, and @ref Ge varies across architectures.
 * On @c x86, they perform non-signaling comparisons,
 * while the others perform signaling comparisons.
 *
 * Signaling comparisons raise an invalid floating-point exception
 * if any lane of the corresponding lanes is set to @c NaN.
 * It is recommended to use @ref kCMPSignal to detect this behavior when needed.
 *
 * @rst
 * Index
 * -----
 * @endrst
 *
 * | Name                  |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
 * |-----------------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
 * | @ref Gt               | x | x | x | x | x | x | x | x | x | x |
 * | @ref Ge               | x | x | x | x | x | x | x | x | x | x |
 * | @ref Lt               | x | x | x | x | x | x | x | x | x | x |
 * | @ref Le               | x | x | x | x | x | x | x | x | x | x |
 * | @ref Eq               | x | x | x | x | x | x | x | x | x | x |
 * | @ref Ne               | x | x | x | x | x | x | x | x | x | x |
 * | @ref NotNan           | - | - | - | - | - | - | - | - | x | x |
 * | @ref Any              | x | x | x | x | x | x | x | x | x | x |
 * | @ref All              | x | x | x | x | x | x | x | x | x | x |
 *
 * @rst
 * Definitions
 * -----------
 * @endrst
 * @{
 */
/***************************
 * Comparison
 ***************************/
/// Greater than.
/// Performs a greater than comparison between two N-lane vectors and returns a mask.
template<typename TVec, typename TMaskRet = Mask<GetLaneType<TVec>>>
NPY_FINLINE TMaskRet Gt(const TVec &a, const TVec &b);
/// Greater than or equal.
/// Performs a greater than or equal comparison between two N-lane vectors and returns a mask.
template<typename TVec, typename TMaskRet = Mask<GetLaneType<TVec>>>
NPY_FINLINE TMaskRet Ge(const TVec &a, const TVec &b);
/// Less than.
/// Performs a less than comparison between two N-lane vectors and returns a mask.
template<typename TVec, typename TMaskRet = Mask<GetLaneType<TVec>>>
NPY_FINLINE TMaskRet Lt(const TVec &a, const TVec &b);
/// Less than or equal.
/// Performs a less than or equal comparison between two N-lane vectors and returns a mask.
template<typename TVec, typename TMaskRet = Mask<GetLaneType<TVec>>>
NPY_FINLINE TMaskRet Le(const TVec &a, const TVec &b);
/// Equal.
/// Performs an equal comparison between two N-lane vectors and returns a mask.
template<typename TVec, typename TMaskRet = Mask<GetLaneType<TVec>>>
NPY_FINLINE TMaskRet Eq(const TVec &a, const TVec &b);
/// Not Equal.
/// Performs a not equal comparison between two N-lane vectors and returns a mask.
/// @note This intrinsic performs an unordered comparison.
template<typename TVec, typename TMaskRet = Mask<GetLaneType<TVec>>>
NPY_FINLINE TMaskRet Ne(const TVec &a, const TVec &b);
/**
 * Test a special case where the floating-point data types are not NaN.
 * It performs a comparison operation on each lane of the input vector,
 * returning a mask where each bit/lane is true if the corresponding
 * element is not NaN.
 *
 * @tparam TVec The N-lane vector data type
 * @tparam TMaskRet The mask data type with the same lane type as TVec
 * @param a The input N-lane vector
 * @return A mask where each lane/bit is @c true if the corresponding
 *         element is not NaN, @c false otherwise
 */
template<typename TVec, typename TMaskRet = Mask<GetLaneType<TVec>>>
NPY_FINLINE TMaskRet NotNan(const TVec &a);
/**
 * All not equal to zero.
 *
 * Check if all lanes in the vector or all the bits in the mask are true,
 * where true is interpreted as a non-zero value.
 *
 * @tparam TVecOrMask The N-lane vector or mask data type
 * @param a The input N-lane vector or mask
 * @return @c true if all lanes in the vector or
 *         all bits in the mask are true, @c false otherwise
 */
template<typename TVecOrMask>
NPY_FINLINE bool All(const TVecOrMask &a);
/**
 * Any not equal to zero.
 *
 * Check if any lane in the vector or any bit in the mask is true,
 * where true is interpreted as a non-zero value.
 *
 * @tparam TVecOrMask The N-lane vector or mask data type
 * @param a The input N-lane vector
 * @return @c true if any lanes in the vector or
 *         any bit in the mask is true, @c false otherwise
 */
template<typename TVecOrMask>
NPY_FINLINE bool Any(const TVecOrMask &a);
/// @}

/**@addtogroup cpp_simd_conversion Conversion
 * @ingroup cpp_simd
 * @rst
 * Index
 * -----
 * @endrst
 *
 *
 * | Name                |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
 * |:--------------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
 * | @ref ToMask         | x | x | x | x | x | x | x | x | - | - |
 * | @ref ToVec          | x | x | x | x | x | x | x | x | - | - |
 * | @ref Expand         | x | - | x | - | - | - | - | - | - | - |
 * | @ref Pack           | x |   | x |   | x |   | x |   |   |   |
 * @{
 */
/**
 * Convert an N-lane vector to a mask representation.
 *
 * It creates a mask where each lane is set to true if the corresponding lane
 * in the input vector is all bits populated, and false if all bits are set to zero.
 * The behavior is undefined if the input vector contains lanes with mixed bit patterns.
 *
 * For example:
 * @code
 * ToMask(Set(int32_t(-1), 0, -1, 0))
 * ----------------------------------
 * 0b1010...
 * @endcode
 *
 * @tparam TVec The N-lane vector data type
 * @tparam TMaskRet The mask data type with the same lane type as TVec
 * @param a The input N-lane vector
 * @return A mask representation where each bit is set to true or false based on the input vector.
 */
template<typename TVec, typename TMaskRet = Mask<GetLaneType<TVec>>>
NPY_FINLINE TMaskRet ToMask(const TVec &a);
/**
 * Convert a mask to a vector representation.
 *
 * It converts a mask to a vector representation where each lane in the vector
 * is set to the maximum value of the corresponding lane in the input mask.
 *
 * @tparam TLane The lane type of the output vector its required due to nature of the type Mask
 * @param a The input mask
 * @param tag An optional tag value for specifying the lane type of the output vector
 * @return A vector representation where each lane is set to the maximum
 *         value of the corresponding lane in the input mask
 */
template<typename TLane, typename TMask = Mask<TLane>>
NPY_FINLINE Vec<TLane> ToVec(const TMask &a, TLane tag = 0);
// template<typename TLane>
// NPY_FINLINE uint64_t ToBits(const Mask<TLane> &a, TLane tag = 0);
/**
 * Expand a vector to a wider vector representation.
 *
 * This intrinsic expands a vector by doubling it to a wider vector representation,
 * with doubled lane type.
 * Each lane of the lower and higher halves of the input vector is duplicated across
 * multiple lanes in a tuple of two N-lane vectors.
 *
 * @tparam TVec The N-lane vector data type
 * @tparam TVec2Ret The wider tuple data type for the output deduced from TVec.
 * @param a The input vector, only Vec<uint8_t> and Vec<uint16_t> supported for now.
 * @return A tuple of two N-lane vectors with double width lane type.
 *
 * @par Example:
 * @code{.cpp}
 * // Assuming the number for Vec<uint16_t> is eight (128-bit width register)
 * Vec<uint16_t> vec = Set(uint16_t(1), 2, 3, 4, 5, 6, 7, 8);
 * // Tuple of 2 vectors of double type of uint16_t
 * Vec2<uint32_t> ret = Expand(vec);
 * GetTuple<0>(ret);  // Result of the first vector of the tuple: Vec<uint32_t>(1, 2, 3, 4)
 * GetTuple<1>(ret);  // Result of the second vector of the tuple: Vec<uint32_t>(5, 6, 7, 8)
 * @endcode
 */
template<typename TVec, typename TVec2Ret = Vec2<DoubleIt<GetLaneType<TVec>>>>
NPY_FINLINE TVec2Ret Expand(const TVec &a);
/**
 * Pack two masks into a single mask with halved lane type.
 *
 * This intrinsic packs two masks into a single mask where the lane type is halved.
 * The resulting mask is created by interleaving the whole lanes from the input masks.
 *
 * @tparam TLane The lane type of the input Mask, required due to nature of the type Mask
 * @tparam TMask The mask data type deduced from TLane
 * @param a The first input mask
 * @param b The second input mask
 * @param tag An optional tag value
 * @return A mask with halved lane type.
 *
 * @par Example:
 * @code{.cpp}
 * // Set even truth: 0b1010...
 * Mask<int16_t> a = ToMask(Set(int16_t(-1), 0));
 * // Set odd truth: 0b0101
 * Mask<int16_t> b = ToMask(Set(int16_t(0), int16_t(-1)));
 * Pack<uint8_t>(a, b); // Result of the pack: 0b1010...0101...
 * @endcode
 */
template<typename TLane, typename TMask=Mask<TLane>>
NPY_FINLINE Mask<HalveIt<TLane>> Pack(const TMask &a, const TMask &b, TLane tag = 0);

/// Pack four 32-bit mask into one 8-bit mask
NPY_FINLINE Mask<uint8_t> Pack(const Mask<uint32_t> &a, const Mask<uint32_t> &b,
                               const Mask<uint32_t> &c, const Mask<uint32_t> &d);

/// Pack eight 64-bit mask into one 8-bit mask
NPY_FINLINE Mask<uint8_t> Pack(const Mask<uint64_t> &a, const Mask<uint64_t> &b,
                               const Mask<uint64_t> &c, const Mask<uint64_t> &d,
                               const Mask<uint64_t> &e, const Mask<uint64_t> &f,
                               const Mask<uint64_t> &g, const Mask<uint64_t> &h);
/// @}


/**@addtogroup cpp_simd_arithmetic Arithmetic
 * @ingroup cpp_simd
 * @rst
 * Index
 * -----
 * @endrst
 * @{
 *
 * | Name                    |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
 * |-------------------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
 * | @ref Add                | x | x | x | x | x | x | x | x | x | x |
 * | @ref IfAdd              | x | x | x | x | x | x | x | x | x | x |
 * | @ref Adds               | x | x | x | x | - | - | - | - | - | - |
 * | @ref Sub                | x | x | x | x | x | x | x | x | x | x |
 * | @ref IfSub              | x | x | x | x | x | x | x | x | x | x |
 * | @ref Subs               | x | x | x | x | - | - | - | - | - | - |
 * | @ref Mul                | x | x | x | x | x | x | - | - | x | x |
 * | @ref Div                | - | - | - | - | - | - | - | - | x | x |
 * | @ref Divisor            | x | x | x | x | x | x | x | x | - | - |
 * | @ref Div "Div (const)"  | x | x | x | x | x | x | x | x | - | - |
 * | @ref MulAdd             | - | - | - | - | - | - | - | - | x | x |
 * | @ref MulSub             | - | - | - | - | - | - | - | - | x | x |
 * | @ref NegMulAdd          | - | - | - | - | - | - | - | - | x | x |
 * | @ref NegMulSub          | - | - | - | - | - | - | - | - | x | x |
 * | @ref MulAddSub          | - | - | - | - | - | - | - | - | x | x |
 * | @ref Sum                | - | - | - | - | x | - | x | - | x | x |
 * | @ref Sumup              | x | - | x | - | - | - | - | - | - | - |
 *
 * @rst
 * Definitions
 * -----------
 * @endrst
 */
/**
 * Add two vectors element-wise.
 *
 * This intrinsic performs element-wise addition of two vectors and returns the result.
 *
 * @tparam TVec The N-lane vector data type
 * @param a The first input vector
 * @param b The second input vector
 * @return The element-wise sum of the input vectors
 */
template<typename TVec>
NPY_FINLINE TVec Add(const TVec &a, const TVec &b);
/**
 * Conditional addition of vectors based on a mask.
 *
 * This intrinsic performs conditional addition of vectors based on a mask.
 * It adds the corresponding lanes of vectors `a` and `b` if the corresponding
 * mask element is true, and sets the corresponding lanes of vector `c`
 * if the corresponding mask element is false.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:NLanes] = m ? a + b : c
 * @endcode
 *
 * @tparam TVec The N-lane vector data type
 * @tparam TMask The mask data type deduced from `TVec`
 * @param m The mask determining the addition operation
 * @param a The first input vector
 * @param b The second input vector
 * @param c The third input vector
 * @return The result of the conditional addition based on the mask
 */
template<typename TVec, typename TMask = Mask<GetLaneType<TVec>>>
NPY_FINLINE TVec IfAdd(const TMask &m, const TVec &a, const TVec &b, const TVec &c);
/**
 * Add two vectors with saturation.
 *
 * This intrinsic performs element-wise addition of two vectors with saturation,
 * meaning that any overflow or underflow is clamped to the maximum or minimum value
 * representable by the vector's lane type.
 *
 * @tparam TVec The N-lane vector data type
 * @param a The first input vector
 * @param b The second input vector
 * @return The element-wise sum of the input vectors with saturation
 */
template<typename TVec>
NPY_FINLINE TVec Adds(const TVec &a, const TVec &b);
/**
 * Subtract two vectors element-wise.
 *
 * This intrinsic performs element-wise subtraction of two vectors and returns the result.
 *
 * @tparam TVec The N-lane vector data type
 * @param a The first input vector
 * @param b The second input vector
 * @return The element-wise difference of the input vectors
 */
template<typename TVec>
NPY_FINLINE TVec Sub(const TVec &a, const TVec &b);
/**
 * Conditional subtraction of vectors based on a mask.
 *
 * This intrinsic performs conditional subtraction of vectors based on a mask.
 * It subtracts the corresponding lanes of vectors `a` and `b` if the corresponding
 * mask element is true, and sets the corresponding lanes of vector `c`
 * if the corresponding mask element is false.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:NLanes] = m ? a - b : c
 * @endcode
 *
 * @tparam TVec The N-lane vector data type
 * @tparam TMask The mask data type deduced from `TVec`
 * @param m The mask determining the subtraction operation
 * @param a The first input vector
 * @param b The second input vector
 * @param c The third input vector
 * @return The result of the conditional subtraction based on the mask
 */
template<typename TVec, typename TMask = Mask<GetLaneType<TVec>>>
NPY_FINLINE TVec IfSub(const TMask &m, const TVec &a, const TVec &b, const TVec &c);
/**
 * Subtract two vectors with saturation.
 *
 * This intrinsic performs element-wise subtraction of two vectors with saturation,
 * meaning that any overflow or underflow is clamped to the maximum or minimum value
 * representable by the vector's lane type.
 *
 * @tparam TVec The N-lane vector data type
 * @param a The first input vector
 * @param b The second input vector
 * @return The element-wise difference of the input vectors with saturation
 */
template<typename TVec>
NPY_FINLINE TVec Subs(const TVec &a, const TVec &b);
/**
 * Multiply two vectors element-wise.
 *
 * This intrinsic performs element-wise multiplication of two vectors and returns the result.
 *
 * @tparam TVec The N-lane vector data type
 * @param a The first input vector
 * @param b The second input vector
 * @return The element-wise product of the input vectors
 */
template<typename TVec>
NPY_FINLINE TVec Mul(const TVec &a, const TVec &b);
/**
 * Divide two vectors element-wise.
 *
 * This intrinsic performs element-wise division of two vectors and returns the result.
 *
 * @tparam TVec The N-lane vector data type
 * @tparam TLane The lane data type deduced from TVec
 * @param a The first input vector
 * @param b The second input vector
 * @return The element-wise division of the input vectors
 */
template<typename TVec, typename TLane = GetLaneType<TVec>>
NPY_FINLINE TVec Div(const TVec &a, const Vec<TLane> &b);
/**
 * Divide each lane in a vector by a scalar value.
 *
 * Performs element-wise integer division between the lanes of the input vector 'a'
 * and the scalar value `b`. The scalar value `b` is a tuple of three N-lane vectors
 * that contain the precomputed parameters of the divisor. These parameters must be generated
 * using the np::simd::Divisor function.
 *
 * The division is performed using the technique of multiplying with the precomputed reciprocal.
 * The implementation is based on the method described by T. Granlund and P. L. Montgomery
 * in their paper "Division by invariant integers using multiplication"
 * (see [Figure 4.1, 5.1](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.1.2556)).
 * This technique shows good performance across different architectures, especially on X86.
 * However, computing the divisor parameters is relatively expensive, so this implementation
 * is most effective when the divisor is a scalar and used multiple times.
 *
 * Example:
 * @code
 * size_t nlanes = NLanes<int32_t>();            // number of lanes
 * Vec3<int32_t> x = Divisor(int32_t(0x6e70));   // initialize divisor params
 * for (; len >= nlanes; src += nlanes, dst += nlanes, len -= nlanes) {
 *     Vec<int32_t> a = Load(src);   // load int32_t vector from memory
 *     Vec<int32_t> div = Div(a, x); // divide all elements by x
 *     Store(dst, a);                // store int32_t vector into memory
 * }
 * @endcode
 *
 * @tparam TVec The N-lane vector data type
 * @tparam TLane The lane data type deduced from TVec
 * @param a The input vector
 * @param b A tuple of three N-lane vectors that contain the precomputed
 *          parameters of the divisor
 * @return The resulting vector after division
 */
template<typename TVec, typename TLane = GetLaneType<TVec>>
NPY_FINLINE TVec Div(const TVec &a, const Vec3<TLane> &b);
/**
 * Generate the precomputed parameters of the divisor.
 *
 * This intrinsic generates a tuple of three N-lane vectors that contain
 * the precomputed parameters of the divisor. These parameters are used
 * in the np::simd::Div function to perform efficient element-wise
 * division of a vector by a scalar value.
 *
 * Example:
 * @code
 * Vec3<int32_t> x = Divisor(int32_t(0x6e70)); // Generate divisor params
 * Vec<int32_t> a = ...;                      // Input vector
 * Vec<int32_t> div = Div(a, x);              // Divide elements by x
 * @endcode
 *
 * @tparam TLane The data type of the divisor
 * @param d The divisor value
 * @return A tuple of three N-lane vectors containing the precomputed
 *         parameters of the divisor
 */
template<typename TLane>
NPY_FINLINE Vec3<TLane> Divisor(TLane d);
/**
 * Multiply the lanes of two vectors and add the lanes of a third vector.
 *
 * This intrinsic performs element-wise multiplication between the lanes of vectors `a` and `b`,
 * and then adds the lanes of vector `c` in a single step. If the enabled SIMD extension supports
 * native fused multiply-add (FMA) instructions, the operation may be performed with a single
 * rounding. The support for FMA can be checked using the constant @c np::simd::kSupportFMA.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:NLanes] = a * b + c;
 * @endcode
 *
 * Example:
 * @code
 * Vec<float> a = ...;   // First input vector
 * Vec<float> b = ...;   // Second input vector
 * Vec<float> c = ...;   // Third input vector
 * Vec<float> result = MulAdd(a, b, c);   // Multiply and add vectors
 * @endcode
 *
 * @tparam TVec The N-lane vector data type
 * @param a The first input vector
 * @param b The second input vector
 * @param c The third input vector
 * @return The resulting vector after multiplication and addition
 */
template<typename TVec>
NPY_FINLINE TVec MulAdd(const TVec &a, const TVec &b, const TVec &c);
/**
 * Multiply the lanes of two vectors and subtract the lanes of a third vector.
 *
 * Similar to np::simd::MulAdd but subtract the lanes of a third vector instead.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:NLanes] = a * b - c;
 * @endcode
 */
template<typename TVec>
NPY_FINLINE TVec MulSub(const TVec &a, const TVec &b, const TVec &c);
/**
 * Negated multiply and add operation.
 *
 * Negates element-wise the product of vector `a`, vector `b`,
 * and then adds the lanes of vector `c`. Similar to np::simd::MulAdd
 * the rounding affected by FMA support and can be checked using
 * the constant @c np::simd::kSupportFMA.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:NLanes] = -(a * b) + c
 * @endcode
 */
template<typename TVec>
NPY_FINLINE TVec NegMulAdd(const TVec &a, const TVec &b, const TVec &c);
/**
 * Negated multiply and subtract operation.
 *
 * Negates element-wise the product of vector `a`, vector `b`,
 * and then subtracts the lanes of vector `c`. Similar to np::simd::MulAdd
 * the rounding affected by FMA support and can be checked using
 * the constant @c np::simd::kSupportFMA.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:NLanes] = -(a * b) - c
 * @endcode
 */
template<typename TVec>
NPY_FINLINE TVec NegMulSub(const TVec &a, const TVec &b, const TVec &c);
/**
 * Multiply the lanes of two vectors, add the lanes of a third vector for odd lanes,
 * and subtract for even lanes.
 *
 * Similar to np::simd::MulAdd the rounding affected by FMA support and
 * can be checked using the constant @c np::simd::kSupportFMA.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:NLanes:2] = a * b + c
 * Vec<TLane>[1:NLanes:2] = a * b - c
 * @endcode
 */
template<typename TVec>
NPY_FINLINE TVec MulAddSub(const TVec &a, const TVec &b, const TVec &c);
/**
 * Compute the sum of lanes in a vector.
 *
 * Computes the sum of all lanes in the input vector `a` and returns the result
 * as a scalar value of lane type.
 */
template<typename TVec, typename TLane = GetLaneType<TVec>>
NPY_FINLINE TLane Sum(const TVec &a);
/**
 * Compute the sum of lanes in a vector and return the result as a widened lane type.
 *
 * Computes the sum of all lanes in the input vector `a` and returns the result
 * as a scalar value of type `TDoubleLane`, which is a widened lane type.
 */
template<typename TVec, typename TDoubleLane = DoubleIt<GetLaneType<TVec>>>
NPY_FINLINE TDoubleLane Sumup(const TVec &a);
/// @}

/**@addtogroup cpp_simd_math Math
 * @ingroup cpp_simd
 * @rst
 * Index
 * -----
 * @endrst
 *
 * | Name                   |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
 * |:-----------------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
 * | @ref Round             |   |   |   |   |   |   |   |   | x | x |
 * | @ref Roundi            |   |   |   |   |   |   |   |   | x | x |
 * | @ref Ceil              |   |   |   |   |   |   |   |   | x | x |
 * | @ref Trunc             |   |   |   |   |   |   |   |   | x | x |
 * | @ref Floor             |   |   |   |   |   |   |   |   | x | x |
 * | @ref Sqrt              | - | - | - | - | - | - | - | - | x | x |
 * | @ref Recip             | - | - | - | - | - | - | - | - | x | x |
 * | @ref Abs               | - | - | - | - | - | - | - | - | x | x |
 * | @ref Min               | x | x | x | x | x | x | x | x | x | x |
 * | @ref Max               | x | x | x | x | x | x | x | x | x | x |
 * | @ref MinProp           |   |   |   |   |   |   |   |   | x | x |
 * | @ref MaxProp           |   |   |   |   |   |   |   |   | x | x |
 * | @ref MinPropNan        |   |   |   |   |   |   |   |   | x | x |
 * | @ref MaxPropNan        |   |   |   |   |   |   |   |   | x | x |
 * | @ref ReduceMin         | x | x | x | x | x | x | x | x | x | x |
 * | @ref ReduceMax         | x | x | x | x | x | x | x | x | x | x |
 * | @ref ReduceMinProp     |   |   |   |   |   |   |   |   | x | x |
 * | @ref ReduceMaxProp     |   |   |   |   |   |   |   |   | x | x |
 * | @ref ReduceMinPropNan  |   |   |   |   |   |   |   |   | x | x |
 * | @ref ReduceMaxPropNan  |   |   |   |   |   |   |   |   | x | x |
 *
 * @rst
 * Definitions
 * -----------
 * @endrst
 * @{
 */
/**
 * Round the lanes of a floating-point N-lane vector to the nearest integral value.
 *
 * The rounding direction may affected by the floating-point environment,
 * such as the @c MXCSR register on @c x86, if the enabled SIMD extension does
 * not provide an operand to set the rounding mode on the fly,
 * otherwise it rounds to nearest even.
 *
 * @note
 * It may raise an @c inexact fp exception if the value returned differs in value
 * from the input N-lane vector.
 */
template<typename TVec>
NPY_FINLINE TVec Round(const TVec &a);
/**
 * Truncate the lanes of a floating-point N-lane vector towards zero.
 *
 * @note
 * It may raise an inexact fp exception if the value returned differs in value
 * from the input N-lane vector.
 */
template<typename TVec>
NPY_FINLINE TVec Trunc(const TVec &a);
/**
 * Truncate the lanes of a floating-point N-lane vector towards positive infinity.
 *
 * @note
 * It may raise an inexact fp exception if the value returned differs in value
 * from the input N-lane vector.
 */
template<typename TVec>
NPY_FINLINE TVec Ceil(const TVec &a);
/**
 * Truncate the lanes of a floating-point N-lane vector towards negative infinity.
 *
 * @note
 * It may raise an inexact fp exception if the value returned differs in value
 * from the input N-lane vector.
 */
template<typename TVec>
NPY_FINLINE TVec Floor(const TVec &a);
/**
 * Round the lanes of a floating-point N-lane vector to the nearest integral value.
 *
 * Similar to np::simd::Round, except it returns a signed integer N-lane vector
 * that match the size of the lane type of the input vector.
 */
template<typename TVec, typename TVecRet = Vec<MakeSigned<GetLaneType<TVec>>>>
NPY_FINLINE TVecRet Roundi(const TVec &a);
/**
 * Round the lanes of a floating-point N-lane vector to the nearest integer value.
 *
 * Similar to np::simd::Round, except it returns a halved signed integer N-lane vector
 * after rounding and packing.
 *
 * @par Scheme:
 * @code
 * Vec<HalveIt<MakeSigned<TLane>>> = pack(int(round(a), int(round(b)))
 * @enddcode
 *
 * @par Example:
 * @code{.cpp}
 * Vec<int32_t> roundi = Roundi(Set(1.5), Set(2.5));
 * // result: {1, ..., 2, ...}
 * @endcode
 */
template<typename TVec, typename TVecRet = Vec<HalveIt<MakeSigned<GetLaneType<TVec>>>>>
NPY_FINLINE TVecRet Roundi(const TVec &a, const TVec &b);
/**
 * Calculate the square root of each lane in a vector.
 *
 * Calculates the square root of each lane in the input vector `a` and returns
 * a new vector with the square root values.
 */
template<typename TVec>
NPY_FINLINE TVec Sqrt(const TVec &a);
/**
 * Calculate the reciprocal of each lane in a vector.
 *
 * Calculates the reciprocal (1/x) of each lane in the input vector `a` and returns
 * a new vector with the reciprocal values.
 */
template<typename TVec>
NPY_FINLINE TVec Recip(const TVec &a);
/**
 * Calculate the absolute value of each lane in a vector.
 *
 * Calculates the absolute value of each lane in the input vector `a` and returns
 * a new vector with the absolute values.
 */
template<typename TVec>
NPY_FINLINE TVec Abs(const TVec &a);
/**
 * Maximum.
 *
 * Returns a vector where each lane contains the maximum value between the
 * corresponding lanes of vectors `a` and `b`.
 *
 * @note
 * For floating-point datatypes the NaN values considerd undefined and
 * left it to the nature of the native instructions.
 * Use np::simd::MaxPropNan and np::simd::MaxProp to guarantees the behavior of
 * NaNs.
 */
template<typename TVec>
NPY_FINLINE TVec Max(const TVec &a, const TVec &b);
/**
 * Minimum.
 *
 * Returns a vector where each lane contains the minimum value between the
 * corresponding lanes of vectors `a` and `b`.
 *
 * @note
 * For floating-point datatypes the NaN values considerd undefined and
 * left it to the nature of the native instructions.
 * Use np::simd::MinPropNan and np::simd::MinProp to guarantees the behavior of
 * NaNs.
 */
template<typename TVec>
NPY_FINLINE TVec Min(const TVec &a, const TVec &b);
/**
 * Maximum with Floating-Point with Non-NaN Propagation.
 *
 * Returns a vector where each lane contains the maximum value between the
 * corresponding lanes of vectors `a` and `b`. If one of the values is NaN, the
 * non-NaN value is propagated as the result.
 */
template<typename TVec>
NPY_FINLINE TVec MaxProp(const TVec &a, const TVec &b);
/**
 * Minimum with Floating-Point with Non-NaN Propagation.
 *
 * Returns a vector where each lane contains the minimum value between the
 * corresponding lanes of vectors `a` and `b`. If one of the values is NaN, the
 * non-NaN value is propagated as the result.
 */
template<typename TVec>
NPY_FINLINE TVec MinProp(const TVec &a, const TVec &b);
/**
 * Maximum with Floating-Point with NaN Propagation.
 *
 * Returns a vector where each lane contains the maximum value between the
 * corresponding lanes of vectors `a` and `b`. If one of the values is NaN, the
 * NaN value is propagated as the result.
 */
template<typename TVec>
NPY_FINLINE TVec MaxPropNan(const TVec &a, const TVec &b);
/**
 * Minimum with with NaN Propagation.
 *
 * Returns a vector where each lane contains the minimum value between the
 * corresponding lanes of vectors `a` and `b`. If one of the values is a propagation
 * value (such as NaN or Inf), the propagation value is propagated as the result.
 */
template<typename TVec>
NPY_FINLINE TVec MinPropNan(const TVec &a, const TVec &b);
/**
 * Reduce Maximum.
 *
 * Returns the maximum value among all the lanes in the vector 'a'.
 *
 * @note
 * For floating-point datatypes the NaN values considerd undefined and
 * left it to the nature of the native instructions.
 * Use np::simd::ReduceMaxPropNan and np::simd::ReduceMaxProp to
 * guarantees the behavior of NaNs.
 */
template<typename TVec, typename TLane = GetLaneType<TVec>>
NPY_FINLINE TLane ReduceMax(const TVec &a);
/**
 * Reduce Minimum.
 *
 * Returns the minimum value among all the lanes in the vector 'a'.
 *
 * @note
 * For floating-point datatypes the NaN values considerd undefined and
 * left it to the nature of the native instructions.
 * Use np::simd::ReduceMinPropNan and np::simd::ReduceMinProp to
 * guarantees the behavior of NaNs.
 */
template<typename TVec, typename TLane = GetLaneType<TVec>>
NPY_FINLINE TLane ReduceMin(const TVec &a);
/**
 * Reduce Maximum Floating-Point with Non-NaN Propagation.
 *
 * Returns the maximum value among all the lanes in the vector 'a'. If the vector
 * contains NaN values, the non-NaN maximum value is returned.
 */
template<typename TVec, typename TLane = GetLaneType<TVec>>
NPY_FINLINE TLane ReduceMaxProp(const TVec &a);
/**
 * Reduce Minimum Floating-Point with Non-NaN Propagation.
 *
 * Returns the minimum value among all the lanes in the vector 'a'. If the vector
 * contains NaN values, the non-NaN minimum value is returned.
 */
template<typename TVec, typename TLane = GetLaneType<TVec>>
NPY_FINLINE TLane ReduceMinProp(const TVec &a);
/**
 * Reduce Maximum Floating-Point with NaN Propagation.
 *
 * Returns the maximum value among all the lanes in the vector 'a'. If the vector
 * contains NaN, NaN return.
 */
template<typename TVec, typename TLane = GetLaneType<TVec>>
NPY_FINLINE TLane ReduceMaxPropNan(const TVec &a);
/**
 * Reduce Minimum Floating-Point with NaN Propagation.
 *
 * Returns the minimum value among all the lanes in the vector 'a'. If the vector
 * contains NaN, NaN return.
 */
template<typename TVec, typename TLane = GetLaneType<TVec>>
NPY_FINLINE TLane ReduceMinPropNan(const TVec &a);
/// @}

/**@addtogroup cpp_simd_reorder Reorder
 * @ingroup cpp_simd
 * @rst
 * Index
 * -----
 * @endrst
 *
 * | Name                |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
 * |:--------------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
 * | @ref CombineLow     | x | x | x | x | x | x | x | x | x | x |
 * | @ref CombineHigh    | x | x | x | x | x | x | x | x | x | x |
 * | @ref Combine        | x | x | x | x | x | x | x | x | x | x |
 * | @ref Zip            | x | x | x | x | x | x | x | x | x | x |
 * | @ref Unzip          | x | x | x | x | x | x | x | x | x | x |
 * | @ref Reverse64      | x | x | x | x | x | x |   |   | x |   |
 * | @ref Permute128     | - | - | - | - | x | x | x | x | x | x |
 *
 * @rst
 * Definitions
 * -----------
 * @endrst
 * @{
 */
/**
 * Combine lanes of lower halves of two N-lane vectors.
 *
 * This intrinsic combines the lower half lanes of vectors `a` and `b` into an interleaved vector.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:NLanes/2] = a[0:NLanes/2]
 * Vec<TLane>[NLanes/2:NLanes] = b[0:NLanes/2]
 * @endcode
 */
template<typename TVec>
NPY_FINLINE TVec CombineLow(const TVec &a, const TVec &b);
/**
 * Combine lanes of higher halves of two N-lane vectors.
 *
 * This intrinsic combines the higher half lanes of vectors `a` and `b` into an interleaved vector.
 *
 * @par Scheme:
 * @code
 * Vec<TLane>[0:NLanes/2] = a[NLanes/2:NLanes]
 * Vec<TLane>[NLanes/2:NLanes] = b[NLanes/2:NLanes]
 * @endcode
 */
template<typename TVec>
NPY_FINLINE TVec CombineHigh(const TVec &a, const TVec &b);
/**
 * Combine the lower and higher half lanes into a tuple of two N-lane vectors.
 *
 * This intrinics is equivalent to SetTuple(CombineLower(a, b), CombineHigher(a, b))
 */
template<typename TVec, typename TLane = GetLaneType<TVec>>
NPY_FINLINE Vec2<TLane> Combine(const TVec &a, const TVec &b);
/**
 * Zip two N-lane vectors.
 *
 * This intrinsic interleaves the lanes of vectors `a` and `b`.
 *
 * @par Example:
 * @code{.cpp}
 * Vec2<double> result = Zip(Set(1.0, 2.0), Set(3.0, 4.0));
 * // result = {{1.0, 3.0, ...}, {2.0, 4.0, ...)}}
 * @endcode
 *
 * @return A tuple of two N-lane vectors containing the interleaved result.
 */
template<typename TVec, typename TLane = GetLaneType<TVec>>
NPY_FINLINE Vec2<TLane> Zip(const TVec &a, const TVec &b);
/**
 * De-interleave two N-lane vectors.
 *
 * This intrinsic de-interleaves the lanes of vectors `a` and `b`.
 *
 * @par Example:
 * @code{.cpp}
 * Vec2<double> result = Zip(Set(1.0, 3.0), Set(2.0, 4.0));
 * // result = {{1.0, 2.0, ...}, {3.0, 4.0, ...)}}
 * @endcode
 *
 * @return A tuple of two N-lane vectors containing the de-interleaved result.
 */
template<typename TVec, typename TLane = GetLaneType<TVec>>
NPY_FINLINE Vec2<TLane> Unzip(const TVec &a, const TVec &b);
/**
 * Reverse elements within each 64-bit lane of an N-lane vector.
 *
 * @par Example:
 * @code{.cpp}
 * Vec<int32_t> result = Reverse64(Set(int32_t(2), 1, 3, 4));
 * // result = {1, 2, 3, 4, ...}
 * Vec<int16_t> result = Reverse64(Set(int16_t(4), 3, 2, 1, 8, 7, 6, 5));
 * // result = {1, 2, 3, 4, 5, 6, 7, 8, ...}
 * @endcode
 */
template<typename TVec>
NPY_FINLINE TVec Reverse64(const TVec &a);
/**
 * Permute 32-bit elements within each 128-bit lane of an N-lane vector.
 *
 * This function performs a permutation of 32-bit elements within each 128-bit lane
 * of the input vector `a`, based on the specified indices `L0`, `L1`, `L2`, and `L3`.
 *
 * Example:
 * @code
 * Vec<uint32_t> a = Set(uint32_t(10), 20, 30, 40, 50, 60, 70, 80);   // Input vector
 * Vec<uint32_t> result = Permute128<2, 0, 3, 1>(a);  // Permute elements within each 128-bit lane
 * // result: {30, 10, 40, 20, 70, 50, 80, 60, ...}
 * @endcode
 *
 * @tparam L0 The index of the element from 0 to 3 to be placed at position 0 within each 128-bit lane
 * @tparam L1 The index of the element from 0 to 3 to be placed at position 1 within each 128-bit lane
 * @tparam L2 The index of the element from 0 to 3 to be placed at position 2 within each 128-bit lane
 * @tparam L3 The index of the element from 0 to 3 to be placed at position 3 within each 128-bit lane
 * @param a The input vector
 * @return The resulting vector with permuted elements within each 128-bit lane
 */
template <int L0, int L1, int L2, int L3>
NPY_FINLINE Vec<uint32_t> Permute128(const Vec<uint32_t> &a);
/// @overload
template <int L0, int L1, int L2, int L3>
NPY_FINLINE Vec<int32_t> Permute128(const Vec<int32_t> &a);
/// @overload
template <int L0, int L1, int L2, int L3>
NPY_FINLINE Vec<float> Permute128(const Vec<float> &a);
/**
 * Permute 64-bit elements within each 128-bit lane of an N-lane vector.
 *
 * This function performs a permutation of 64-bit elements within each 128-bit lane
 * of the input vector `a`, based on the specified indices `L0`, `L1`.
 *
 * Example:
 * @code
 * Vec<uint32_t> a = Set(uint64_t(10), 20, 30, 40);   // Input vector
 * Vec<uint32_t> result = Permute128<1, 0, 1, 0>(a);  // Permute elements within each 128-bit lane
 * // result: {20, 10, 40, 30, ...}
 * @endcode
 *
 * @tparam L0 The index of the element from 0 to 1 to be placed at position 0 within each 128-bit lane
 * @tparam L1 The index of the element from 0 to 1 to be placed at position 1 within each 128-bit lane
 * @param a The input vector
 * @return The resulting vector with permuted elements within each 128-bit lane
 */
template <int L0, int L1>
NPY_FINLINE Vec<uint64_t> Permute128(const Vec<uint64_t> &a);
/// @overload
template <int L0, int L1>
NPY_FINLINE Vec<int64_t> Permute128(const Vec<int64_t> &a);
/// @overload
template <int L0, int L1>
NPY_FINLINE Vec<double> Permute128(const Vec<double> &a);

/// @}
} // np::simd_ext
#endif // NUMPY_CORE_SRC_COMMON_SIMD_FORWARD_INC_HPP_
