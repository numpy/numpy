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
float32* data = /* ... */;
Vec<float32> v = LoadU(data);
v = Add(v, v);
StoreU(v, data);

// Use np::simd128 for fixed 128-bit SIMD operations
using namespace np::simd128;
Vec<float32> v128 = LoadU(data);
v128 = Add(v128, v128);
StoreU(v128, data);
```

### Checking for SIMD Support

```cpp
#include "simd/simd.hpp"

// Check if SIMD is enabled
#if NPY_SIMDX
    // SIMD code
#else
    // Scalar fallback code
#endif

// Check for float64 support
#if NPY_SIMDX_F64
    // Use float64 SIMD operations
#endif

// Check for FMA support
#if NPY_SIMDX_FMA
    // Use FMA operations
#endif
```

### Type Support Checks

```cpp
#include "simd/simd.hpp"

// Check if float64 operations are supported
if constexpr (np::simd::kSupportLane<double>) {
    // Use float64 operations
}
```

## Available Operations

The wrapper provides the following common operations that used in NumPy, additional Highway operations can be accessed via then 
`hn` namespace alias inside the `simd` or `simd128` namespaces.

## Build Configuration

The SIMD wrapper automatically disables SIMD operations when optimizations are disabled:

- When `NPY_DISABLE_OPTIMIZATION` is defined, SIMD operations are disabled
- SIMD is enabled only when the Highway target is not scalar (`HWY_TARGET != HWY_SCALAR`)

## Design Notes

1. **Why avoid Highway scalar operations?**
   - NumPy already provides kernels for scalar operations
   - Compilers can better optimize standard library implementations
   - Not all Highway intrinsics are fully supported in scalar mode

2. **Legacy Universal Intrinsics**
   - The older universal intrinsics C interface (in `simd.h` and accessible via `NPY_SIMD` macros) is deprecated
   - All new SIMD code should use this Highway-based wrapper (accessible via `NPY_SIMDX` macros) 
   - The legacy code is maintained for compatibility but will eventually be removed
   
3. **Feature Detection Macros**
   - Always use the NumPy-specific macros (`NPY_SIMDX_F16`, `NPY_SIMDX_F64`, `NPY_SIMDX_FMA`) rather than Highway macros
   - These macros not only check for hardware capabilities but also verify that SIMD optimization is enabled
   - This prevents unexpected behavior when optimization is disabled

2. **Namespace Design**
   - `np::simd`: Maximum width SIMD operations (scalable)
   - `np::simd128`: Fixed 128-bit SIMD operations
   - `hn`: Highway namespace alias (available within the SIMD namespaces)

3. **Template Type Parameters**
   - `TLane`: The scalar type for each vector lane (e.g., uint8_t, float, double)

## Extending

To add more operations from Highway:

Import them in the `simd.inc.hpp` file using the `using` directive or define warpper intrinsic 
if the original intrinsic requires a class tag.
```c

## Requirements

- C++17 or later
- Google Highway library

## License

Same as NumPy's license
