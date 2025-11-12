#!/usr/bin/env python3
"""
Test script for same_value casting functionality in np.can_cast
This demonstrates the new same_value casting support.
"""

import numpy as np
import sys

def test_can_cast_same_value():
    """Test that can_cast supports same_value casting parameter"""
    
    print("Testing same_value casting support in np.can_cast")
    
    # Test cases where same_value casting should return True
    test_cases_true = [
        # Same types should always work
        (np.int32, np.int32, "Identity casting"),
        (np.float64, np.float64, "Identity casting"),
        
        # Widening integer casts that preserve values
        (np.int8, np.int16, "Widening integer cast"),
        (np.int16, np.int32, "Widening integer cast"),
        (np.int32, np.int64, "Widening integer cast"),
        
        # Widening unsigned integer casts
        (np.uint8, np.uint16, "Widening unsigned cast"),
        (np.uint16, np.uint32, "Widening unsigned cast"),
        
        # Integer to larger float (should preserve exact values for reasonable ranges)
        (np.int8, np.float64, "Integer to double precision"),
        (np.int16, np.float64, "Integer to double precision"),
    ]
    
    # Test cases where same_value casting should return False  
    test_cases_false = [
        # Narrowing casts that could lose data
        (np.int64, np.int32, "Narrowing integer cast"),
        (np.float64, np.float32, "Narrowing float cast"),
        
        # Signed to unsigned (could lose sign information)
        (np.int32, np.uint32, "Signed to unsigned"),
        
        # Float to integer (could lose decimal part)
        (np.float32, np.int32, "Float to integer"),
        (np.float64, np.int64, "Float to integer"),
        
        # Complex to real (loses imaginary part)
        (np.complex64, np.float32, "Complex to real"),
    ]
    
    print(f"Testing {len(test_cases_true)} cases expected to return True...")
    failures = []
    
    for from_dtype, to_dtype, description in test_cases_true:
        try:
            result = np.can_cast(from_dtype, to_dtype, casting='same_value')
            if not result:
                failures.append(f"FAIL: {description} ({from_dtype} -> {to_dtype}) returned False, expected True")
            else:
                print(f"PASS: {description} ({from_dtype} -> {to_dtype})")
        except Exception as e:
            failures.append(f"ERROR: {description} ({from_dtype} -> {to_dtype}) raised {type(e).__name__}: {e}")
    
    print(f"\nTesting {len(test_cases_false)} cases expected to return False...")
    
    for from_dtype, to_dtype, description in test_cases_false:
        try:
            result = np.can_cast(from_dtype, to_dtype, casting='same_value')
            if result:
                failures.append(f"FAIL: {description} ({from_dtype} -> {to_dtype}) returned True, expected False")
            else:
                print(f"PASS: {description} ({from_dtype} -> {to_dtype})")
        except Exception as e:
            failures.append(f"ERROR: {description} ({from_dtype} -> {to_dtype}) raised {type(e).__name__}: {e}")
    
    # Test that the parameter is properly validated
    print("\nTesting parameter validation...")
    try:
        np.can_cast(np.int32, np.int64, casting='invalid_mode')
        failures.append("FAIL: Invalid casting mode 'invalid_mode' should raise ValueError")
    except ValueError:
        print("PASS: Invalid casting mode properly rejected")
    except Exception as e:
        failures.append(f"ERROR: Invalid casting mode raised {type(e).__name__} instead of ValueError: {e}")
    
    # Summary
    if failures:
        print(f"\n{len(failures)} test failures:")
        for failure in failures:
            print(f"  {failure}")
        return False
    else:
        print(f"\nAll tests passed! same_value casting is working correctly.")
        return True

def test_comparison_with_other_modes():
    """Compare same_value with other casting modes"""
    
    print("\n" + "="*60)
    print("Comparing same_value with other casting modes")
    print("="*60)
    
    test_dtype_pairs = [
        (np.int32, np.int64),
        (np.int64, np.int32), 
        (np.float64, np.float32),
        (np.int32, np.float64),
        (np.float64, np.int32),
    ]
    
    casting_modes = ['no', 'equiv', 'safe', 'same_kind', 'same_value', 'unsafe']
    
    print(f"{'From':<12} {'To':<12} {'no':<6} {'equiv':<6} {'safe':<6} {'same_kind':<10} {'same_value':<10} {'unsafe':<6}")
    print("-" * 70)
    
    for from_dtype, to_dtype in test_dtype_pairs:
        results = []
        for mode in casting_modes:
            try:
                result = np.can_cast(from_dtype, to_dtype, casting=mode)
                results.append('T' if result else 'F')
            except Exception:
                results.append('E')
        
        print(f"{str(from_dtype):<12} {str(to_dtype):<12} {results[0]:<6} {results[1]:<6} {results[2]:<6} {results[3]:<10} {results[4]:<10} {results[5]:<6}")

if __name__ == "__main__":
    print("NumPy same_value casting test for can_cast functionality")
    print("=" * 60)
    
    # Check if same_value is supported
    try:
        np.can_cast(np.int32, np.int32, casting='same_value')
        print("✓ same_value casting parameter is supported!")
    except (ValueError, TypeError) as e:
        print(f"✗ same_value casting parameter is NOT supported: {e}")
        print("This indicates the implementation needs to be completed.")
        sys.exit(1)
    
    success = test_can_cast_same_value()
    test_comparison_with_other_modes()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check implementation.")
        sys.exit(1)