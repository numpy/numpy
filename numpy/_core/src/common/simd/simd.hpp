#ifndef NUMPY_CORE_SRC_COMMON_SIMD_SIMD_HPP_
#define NUMPY_CORE_SRC_COMMON_SIMD_SIMD_HPP_

#include "npy_cpu_dispatch.h"

namespace np::NPY_CPU_DISPATCH_CURFX(simd_ext) {
/**
 * @defgroup cpp_simd Universal Intrinsics (C++)
 *
 * The Universal Intrinsics provides a set of types and functions designed to simplify
 * vectorization of code across different platforms. It enables the development of generic code
 * that can take advantage of common SIMD (Single Instruction, Multiple Data) extensions.
 * Currently, It supports SIMD extensions such as X86 (SSE >= 2, AVX2, AVX512*),
 * PPC64LE (VSX >= 2), S390X (VX, VXE, VXE2), and ARM (NEON, ASIMD).
 *
 * @section quick_start Quick Start:
 *
 * The Universal Intrinsics is primarily developed to serve the interests of the NumPy project.
 * It is not intended for public exposure and resides within the private sources of NumPy under
 * the path `numpy/_core/src/common/simd`.
 *
 * **How to Use:**
 *
 * To use the Universal Intrinsics, include the header file "simd/simd.hpp".
 * It is encapsulated within the namespace `np::simd`.
 * It provides various template types such as `Vec`, `Vec2`, `Vec3`, `Vec4`, and `Mask`,
 * along with a wide range of intrinsics for different operations.
 *
 * - Memory operations: @ref cpp_simd_memory
 * - Bitwise operations: @ref cpp_simd_bitwise
 * - Comparison operations: @ref cpp_simd_comparison
 * - Conversion operations: @ref cpp_simd_conversion
 * - Reordering operations: @ref cpp_simd_reorder
 * - Arithmetic operations: @ref cpp_simd_arithmetic
 * - Math operations: @ref cpp_simd_math
 * - Utility functions and miscellaneous operations: @ref cpp_simd_utils & @ref cpp_simd_misc
 *
 * Here's an example demonstrating the usage of Universal Intrinsics
 * to perform a square operation on an array of floating-point values:
 *
 * @code{.cpp}
 * #include "simd/simd.hpp"
 *
 * inline void SimdSquare(const float *src, float *dst, size_t len)
 * {
 * // Check if Universal Intrinsics are supported by the platform
 * #if NPY_SIMD
 *     // Bring Universal Intrinsics into the scope
 *     using namespace np::simd;
 *     // Check if float is supported by the enabled SIMD extension
 *     if constexpr (kSupportLane<float>) {
 *         // Get the number of lanes
 *         const size_t nlanes = NLanes<float>();
 *         // Initialize the length of the array as a signed type
 *         // outside the loop. So we can handle the remaining
 *         // elements of the array using SIMD.
 *         ptrdiff_t slen = len;
 *         // Loop based on the number of nlanes
 *         for (; slen >= nlanes; slen -= nlanes, src += nlanes,
 *                                dst += nlanes) {
 *             // Load from memory
 *             Vec<float> a = Load(src);
 *             // Perform multiplication
 *             Vec<float> square = Mul(a, a);
 *             // Store the product into memory
 *             Store(dst, square);
 *         }
 *         // Handle the remaining elements using SIMD
 *         for (; slen > 0; slen -= nlanes, src += nlanes,
 *                          dst += nlanes) {
 *             // Partial load from memory
 *             Vec<float> a = LoadTill(src, slen);
 *             // Perform multiplication
 *             Vec<float> square = Mul(a, a);
 *             // Store the product into memory
 *             StoreTill(dst, slen, square);
 *         }
 *     }
 * #else
 *     // Fallback when SIMD isn't supported by the current platform
 *     for (; len > 0; ++src, ++dst, --len) {
 *         const float a = *src;
 *         *dst = a * a;
 *    }
 * #endif // NPY_SIMD
 * }
 * @endcode
 *
 *
 * **How are universal intrinsics implemented?**
 *
 * Universal intrinsics rely on preprocessor identifiers that are either statically
 * dispatched from baseline sources or dynamically dispatched within dispatchable sources.
 * These identifiers map each function to the equivalent platform-specific intrinsic,
 * based on the highest level of SIMD extension support available.
 *
 * In cases where an intrinsic has no direct native support by the Instruction Set Architecture
 * (ISA), universal intrinsics emulate it with minimal possible overhead. When
 * designing new kernels that require implementing new universal intrinsics, we aim to
 * achieve the greatest speed gain without favoring one architecture over another.
 *
 * Here's a simplified example demonstrating how to define a contiguous
 * memory load operation on X86:
 *
 * @code{.cpp}
 * #ifdef NPY_HAVE_AVX512F
 *     template<> inline Vec<float> Load(const float *ptr)
 *     { return Vec<float>(_mm512_loadu_ps((const __m512*)ptr)); }
 * #elif defined(NPY_HAVE_AVX2)
 *     template<> inline Vec<float> Load(const float *ptr)
 *     { return Vec<float>(_mm256_loadu_ps((const __m256*)ptr)); }
 * #elif defined(NPY_HAVE_SSE2)
 *     template<> inline Vec<float> Load(const float *ptr)
 *     { return Vec<float>(_mm_loadu_ps((const __m128*)ptr)); }
 * #endif
 * @endcode
 */
} // namespace np::NPY_CPU_DISPATCH_CURFX(simd_ext)

#if defined(NPY_HAVE_SSE2) || defined(NPY_HAVE_VX) || defined(NPY_HAVE_VSX2) || defined(NPY_HAVE_NEON)
    #include "wrapper/wrapper.hpp"
    // Any new intrinsics should be implemented in C++.
    // According to the roadmap, we suppose to get ride of the C interface
    // once we replace the current SIMD kernels using C++.
    #if defined(NPY_HAVE_AVX512F) && !defined(NPY_SIMD_FORCE_256) && !defined(NPY_SIMD_FORCE_128)
        #include "avx512/avx512.hpp"
    #elif defined(NPY_HAVE_AVX2) && !defined(NPY_SIMD_FORCE_128)
        #include "avx2/avx2.hpp"
    #elif defined(NPY_HAVE_SSE2)
        #include "sse/sse.hpp"
    #endif
    #if defined(NPY_HAVE_VX) || (defined(NPY_HAVE_VSX2) && defined(__LITTLE_ENDIAN__))
        #include "vec/vec.hpp"
    #endif
    #ifdef NPY_HAVE_NEON
        #include "neon/neon.hpp"
    #endif
//#elif
    // any new SIMD extensions should included here
    // we should count only on C++ from now on
#else
    /// SIMD width in bits or 0 if there's no SIMD extension available.
    /// For size-less SIMD extension this value will defined as -1.
    #define NPY_SIMD 0
#endif

namespace np {
namespace simd = NPY_CPU_DISPATCH_CURFX(simd_ext);
} // namespace np
#endif // NUMPY_CORE_SRC_COMMON_SIMD_SIMD_HPP_

