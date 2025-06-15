# NumPy SIMD Wrapper for Highway

This directory contains a lightweight C++ wrapper over Google's [Highway](https://github.com/google/highway) SIMD library, designed specifically for NumPy's needs.

> **Note**: This directory also contains the C interface of universal intrinsics (under `simd.h`) which is no longer supported. The Highway wrapper described in this document should be used instead for all new SIMD code.

## Overview

The wrapper simplifies Highway's SIMD interface by eliminating class tags and using lane types directly, which can be deduced from arguments in most cases. This design makes the SIMD code more intuitive and easier to maintain while still leveraging Highway generic intrinsics.

## Architecture

The wrapper consists of two main headers:

1. `simd.hpp`: The main header that defines namespaces and includes configuration macros
2. `simd.inc.hpp`: Implementation details included by `simd.hpp` multiple times for different namespaces

Additionally, this directory contains legacy C interface files for universal intrinsics (`simd.h` and related files) which are deprecated and should not be used for new code. All new SIMD code should use the Highway wrapper.


## Usage

### Basic Usage

```cpp
#include "simd/simd.hpp"

// Use np::simd for maximum width SIMD operations
using namespace np::simd;
float *data = /* ... */;
Vec<float> v = LoadU(data);
v = Add(v, v);
StoreU(v, data);

// Use np::simd128 for fixed 128-bit SIMD operations
using namespace np::simd128;
Vec<float> v128 = LoadU(data);
v128 = Add(v128, v128);
StoreU(v128, data);
```

### Checking for SIMD Support

```cpp
#include "simd/simd.hpp"

// Check if SIMD is enabled
#if NPY_HWY
    // SIMD code
#else
    // Scalar fallback code
#endif

// Check for float64 support
#if NPY_HWY_F64
    // Use float64 SIMD operations
#endif

// Check for FMA support
#if NPY_HWY_FMA
    // Use FMA operations
#endif
```

## Type Support and Constraints

The wrapper provides type constraints to help with SFINAE (Substitution Failure Is Not An Error) and compile-time type checking:

- `kSupportLane<TLane>`: Determines whether the specified lane type is supported by the SIMD extension.
  ```cpp
  // Base template - always defined, even when SIMD is not enabled (for SFINAE)
  template <typename TLane>
  constexpr bool kSupportLane = NPY_HWY != 0;
  template <>
  constexpr bool kSupportLane<double> = NPY_HWY_F64 != 0;
  ```

- `kMaxLanes<TLane>`: Maximum number of lanes supported by the SIMD extension for the specified lane type.
  ```cpp
  template <typename TLane>
  constexpr size_t kMaxLanes = HWY_MAX_LANES_D(_Tag<TLane>);
  ```

```cpp
#include "simd/simd.hpp"

// Check if float64 operations are supported
if constexpr (np::simd::kSupportLane<double>) {
    // Use float64 operations
}
```

These constraints allow for compile-time checking of which lane types are supported, which can be used in SFINAE contexts to enable or disable functions based on type support.

## Available Operations

The wrapper provides the following common operations that are used in NumPy:

- Vector creation operations:
  - `Zero`: Returns a vector with all lanes set to zero
  - `Set`: Returns a vector with all lanes set to the given value
  - `Undefined`: Returns an uninitialized vector
  
- Memory operations:
  - `LoadU`: Unaligned load of a vector from memory
  - `StoreU`: Unaligned store of a vector to memory
  
- Vector information:
  - `Lanes`: Returns the number of vector lanes based on the lane type
  
- Type conversion:
  - `BitCast`: Reinterprets a vector to a different type without modifying the underlying data
  - `VecFromMask`: Converts a mask to a vector
  
- Comparison operations:
  - `Eq`: Element-wise equality comparison
  - `Le`: Element-wise less than or equal comparison
  - `Lt`: Element-wise less than comparison
  - `Gt`: Element-wise greater than comparison
  - `Ge`: Element-wise greater than or equal comparison
  
- Arithmetic operations:
  - `Add`: Element-wise addition
  - `Sub`: Element-wise subtraction
  - `Mul`: Element-wise multiplication
  - `Div`: Element-wise division
  - `Min`: Element-wise minimum
  - `Max`: Element-wise maximum
  - `Abs`: Element-wise absolute value
  - `Sqrt`: Element-wise square root
  
- Logical operations:
  - `And`: Bitwise AND
  - `Or`: Bitwise OR
  - `Xor`: Bitwise XOR
  - `AndNot`: Bitwise AND NOT (a & ~b)

Additional Highway operations can be accessed via the `hn` namespace alias inside the `simd` or `simd128` namespaces.

## Extending

To add more operations from Highway:

1. Import them in the `simd.inc.hpp` file using the `using` directive if they don't require a tag:
   ```cpp
   // For operations that don't require a tag
   using hn::FunctionName;
   ```

2. Define wrapper functions for intrinsics that require a class tag:
   ```cpp
   // For operations that require a tag
   template <typename TLane>
   HWY_API ReturnType FunctionName(Args... args) {
       return hn::FunctionName(_Tag<TLane>(), args...);
   }
   ```

3. Add appropriate documentation and SFINAE constraints if needed


## Build Configuration

The SIMD wrapper automatically disables SIMD operations when optimizations are disabled:

- When `NPY_DISABLE_OPTIMIZATION` is defined, SIMD operations are disabled
- SIMD is enabled only when the Highway target is not scalar (`HWY_TARGET != HWY_SCALAR`)
  and not EMU128 (`HWY_TARGET != HWY_EMU128`)

## Design Notes

1. **Why avoid Highway scalar operations?**
   - NumPy already provides kernels for scalar operations
   - Compilers can better optimize standard library implementations
   - Not all Highway intrinsics are fully supported in scalar mode
   - For strict IEEE 754 floating-point compliance requirements, direct scalar
     implementations offer more predictable behavior than EMU128

2. **Legacy Universal Intrinsics**
   - The older universal intrinsics C interface (in `simd.h` and accessible via `NPY_SIMD` macros) is deprecated
   - All new SIMD code should use this Highway-based wrapper (accessible via `NPY_HWY` macros) 
   - The legacy code is maintained for compatibility but will eventually be removed
   
3. **Feature Detection Constants vs. Highway Constants**
   - NumPy-specific constants (`NPY_HWY_F16`, `NPY_HWY_F64`, `NPY_HWY_FMA`) provide additional safety beyond raw Highway constants
   - Highway constants (e.g., `HWY_HAVE_FLOAT16`) only check platform capabilities but don't consider NumPy's build configuration
   - Our constants combine both checks:
     ```cpp
     #define NPY_HWY_F16 (NPY_HWY && HWY_HAVE_FLOAT16)
     ```
   - This ensures SIMD features won't be used when:
     - Platform supports it but NumPy optimization is disabled via meson option:
       ```
       option('disable-optimization', type: 'boolean', value: false,
              description: 'Disable CPU optimized code (dispatch,simd,unroll...)')
       ```
     - Highway target is scalar (`HWY_TARGET == HWY_SCALAR`)
   - Using these constants ensures consistent behavior across different compilation settings
   - Without this additional layer, code might incorrectly try to use SIMD paths in scalar mode

4. **Namespace Design**
   - `np::simd`: Maximum width SIMD operations (scalable)
   - `np::simd128`: Fixed 128-bit SIMD operations
   - `hn`: Highway namespace alias (available within the SIMD namespaces)

5. **Why Namespaces and Why Not Just Use Highway Directly?**
   - Highway's design uses class tag types as template parameters (e.g., `Vec<ScalableTag<float>>`) when defining vector types
   - Many Highway functions require explicitly passing a tag instance as the first parameter
   - This class tag-based approach increases verbosity and complexity in user code
   - Our wrapper eliminates this by internally managing tags through namespaces, letting users directly use types e.g. `Vec<float>`
   - Simple example with raw Highway:
     ```cpp
     // Highway's approach
     float *data = /* ... */;
     
     namespace hn = hwy::HWY_NAMESPACE;
     using namespace hn;
     
     // Full-width operations
     ScalableTag<float> df;  // Create a tag instance
     Vec<decltype(df)> v = LoadU(df, data);  // LoadU requires a tag instance
     StoreU(v, df, data);  // StoreU requires a tag instance
     
     // 128-bit operations
     Full128<float> df128;  // Create a 128-bit tag instance
     Vec<decltype(df128)> v128 = LoadU(df128, data);  // LoadU requires a tag instance
     StoreU(v128, df128, data);  // StoreU requires a tag instance
     ```
  
   - Simple example with our wrapper:
     ```cpp
     // Our wrapper approach
     float *data = /* ... */;
     
     // Full-width operations
     using namespace np::simd;
     Vec<float> v = LoadU(data);  // Full-width vector load
     StoreU(v, data);
     
     // 128-bit operations
     using namespace np::simd128;
     Vec<float> v128 = LoadU(data);  // 128-bit vector load
     StoreU(v128, data);
     ```
   
   - The namespaced approach simplifies code, reduces errors, and provides a more intuitive interface
   - It preserves all Highway operations benefits while reducing cognitive overhead

5. **Why Namespaces Are Essential for This Design?**
   - Namespaces allow us to define different internal tag types (`hn::ScalableTag<TLane>` in `np::simd` vs `hn::Full128<TLane>` in `np::simd128`)
   - This provides a consistent type-based interface (`Vec<float>`) without requiring users to manually create tags
   - Enables using the same function names (like `LoadU`) with different implementations based on SIMD width
   - Without namespaces, we'd have to either reintroduce tags (defeating the purpose of the wrapper) or create different function names for each variant (e.g., `LoadU` vs `LoadU128`)

6. **Template Type Parameters**
   - `TLane`: The scalar type for each vector lane (e.g., uint8_t, float, double)


## Requirements

- C++17 or later
- Google Highway library

## License

Same as NumPy's license
