## Fix ThreadSanitizer data race conditions in NumPy core

**Fixes:** #30085

### Summary

This PR addresses multiple ThreadSanitizer data race conditions identified in NumPy's core C code. The fixes ensure thread safety in both GIL-enabled and free-threaded Python execution modes, resolving race conditions that occur during parallel test execution.

### Problem

ThreadSanitizer detected several race conditions in NumPy:

1. **Loop data cache race** in `legacy_array_method.c` - Global static cache accessed concurrently without synchronization
2. **Coercion cache race** in `array_coercion.c` - Global static cache for array type coercion accessed unsafely  
3. **np.nonzero race conditions** in `item_selection.c` - Boolean array fast paths lacked mutation detection

These races manifest during parallel test execution and with pytest-run-parallel, causing potential crashes, data corruption, or spurious test failures.

### Solution

#### 1. Thread-Safe Cache Management
- Added mutex protection for global static caches using cross-version compatible approach:
  - `PyMutex` for Python 3.13+ 
  - `PyThread_type_lock` for older versions
- Follows existing NumPy patterns (similar to `npy_argparse.c`)

#### 2. Enhanced Race Detection  
- Extended `np.nonzero` race detection to cover boolean array optimized paths
- Ensures consistent error reporting: "number of non-zero array elements changed during function execution"
- Protects both 1D and multi-dimensional code paths

#### 3. Proper Integration
- Mutex initialization integrated into module startup (`init_*_mutex()` functions)
- Function declarations added to appropriate headers
- Maintains backward compatibility and performance

### Changes

**Files Modified:**
- `numpy/_core/src/umath/legacy_array_method.c` - Loop data cache thread safety
- `numpy/_core/src/multiarray/array_coercion.c` - Coercion cache thread safety  
- `numpy/_core/src/multiarray/item_selection.c` - Enhanced nonzero race detection
- `numpy/_core/src/common/umathmodule.h` - Function declaration
- `numpy/_core/src/multiarray/array_coercion.h` - Function declaration
- `numpy/_core/src/umath/umathmodule.c` - Mutex initialization
- `numpy/_core/src/multiarray/multiarraymodule.c` - Mutex initialization

**Key Code Changes:**

```c
// Before: Unsafe cache access
if (loop_data_num_cached > 0) {
    loop_data_num_cached--;
    data = loop_data_cache[loop_data_num_cached];  // RACE CONDITION
}

// After: Thread-safe with mutex
LOCK_LOOP_DATA_CACHE;
if (NPY_LIKELY(loop_data_num_cached > 0)) {
    loop_data_num_cached--;
    data = loop_data_cache[loop_data_num_cached];
    UNLOCK_LOOP_DATA_CACHE;
}
```

### Testing

The fixes address race conditions identified in:
- ThreadSanitizer CI runs
- `test_multithreading.py::test_nonzero` - Now properly detects races in boolean arrays
- pytest-run-parallel execution scenarios

**Expected Results:**
- ThreadSanitizer CI should pass without suppressions for fixed race conditions
- Existing multithreading tests continue to pass
- No performance regression (mutex overhead minimal, only during cache operations)

### Backward Compatibility

- ✅ Maintains API compatibility
- ✅ No breaking changes to existing functionality  
- ✅ Cross-version Python support (3.9+ to 3.14+)
- ✅ Compatible with both GIL and free-threaded modes

### Related Issues

This PR resolves the ThreadSanitizer data races mentioned in:
- Current TSAN suppressions in `tools/ci/tsan_suppressions.txt`
- Issue #30085 ThreadSanitizer data races in parallel test execution

### Review Focus Areas

1. **Thread Safety**: Mutex usage patterns and initialization
2. **Performance**: Minimal overhead introduction
3. **Cross-Platform**: Python version compatibility macros
4. **Integration**: Proper initialization during module startup

The implementation follows NumPy's established patterns for thread-safe code and should resolve all identified ThreadSanitizer race conditions while maintaining full compatibility.