#!/usr/bin/env python3
"""
Test script to verify ThreadSanitizer race condition fixes.
This script tests the race conditions that were fixed:
1. Loop data cache race condition
2. Coercion cache race condition  
3. np.nonzero race condition
"""

import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import warnings

def test_nonzero_race_detection():
    """Test that np.nonzero properly detects array mutations during execution."""
    print("Testing np.nonzero race condition detection...")
    
    # Create an array that will be modified during nonzero operations
    x = np.random.randint(4, size=100).astype(int)
    errors_caught = 0
    total_tests = 0
    
    def modifier_thread():
        """Continuously modify the array"""
        for _ in range(100):
            x[::2] = np.random.randint(2)
            time.sleep(0.001)  # Small delay to allow race conditions
    
    def nonzero_thread():
        """Try to call nonzero and catch race condition errors"""
        nonlocal errors_caught, total_tests
        for _ in range(20):
            try:
                _ = np.nonzero(x)
                total_tests += 1
            except RuntimeError as e:
                if "number of non-zero array elements changed" in str(e):
                    errors_caught += 1
                    total_tests += 1
                else:
                    print(f"Unexpected error: {e}")
            time.sleep(0.001)
    
    # Run modifier and nonzero operations concurrently
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Start modifier thread
        modifier_future = executor.submit(modifier_thread)
        
        # Start multiple nonzero threads
        nonzero_futures = [executor.submit(nonzero_thread) for _ in range(4)]
        
        # Wait for all to complete
        modifier_future.result()
        for future in nonzero_futures:
            future.result()
    
    print(f"Race condition errors caught: {errors_caught}/{total_tests}")
    print(f"Detection rate: {errors_caught/total_tests*100:.1f}%")
    
    if errors_caught > 0:
        print("✓ np.nonzero race condition detection is working!")
        return True
    else:
        print("⚠ No race conditions detected (might be due to timing)")
        return False

def test_coercion_cache_threading():
    """Test that array coercion cache works correctly under threading."""
    print("\nTesting coercion cache threading safety...")
    
    def create_arrays():
        """Create arrays that trigger coercion cache usage"""
        for _ in range(100):
            # These operations trigger array coercion
            arr1 = np.array([[1, 2], [3, 4]])
            arr2 = np.array([5, 6, 7, 8]).reshape(2, 2)
            result = arr1 + arr2
            assert result.shape == (2, 2)
    
    # Run multiple threads creating arrays simultaneously
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(create_arrays) for _ in range(8)]
        
        try:
            for future in futures:
                future.result(timeout=10)  # 10 second timeout
            print("✓ Coercion cache threading test passed!")
            return True
        except Exception as e:
            print(f"✗ Coercion cache threading test failed: {e}")
            return False

def test_ufunc_loop_cache_threading():
    """Test that ufunc loop cache works correctly under threading."""
    print("\nTesting ufunc loop cache threading safety...")
    
    def compute_ufuncs():
        """Perform ufunc operations that use the loop cache"""
        for _ in range(100):
            a = np.random.random(50)
            b = np.random.random(50)
            
            # These operations use the ufunc loop cache
            np.add(a, b)
            np.multiply(a, b)
            np.subtract(a, b)
            np.divide(a, b)
            np.power(a, 2)
            np.sqrt(a)
            np.sin(a)
            np.cos(a)
    
    # Run multiple threads performing ufunc operations
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(compute_ufuncs) for _ in range(8)]
        
        try:
            for future in futures:
                future.result(timeout=10)  # 10 second timeout
            print("✓ Ufunc loop cache threading test passed!")
            return True
        except Exception as e:
            print(f"✗ Ufunc loop cache threading test failed: {e}")
            return False

def main():
    """Run all threading safety tests."""
    print("ThreadSanitizer Race Condition Fix Tests")
    print("=" * 45)
    
    # Suppress warnings to keep output clean
    warnings.filterwarnings('ignore')
    
    results = []
    
    # Test 1: np.nonzero race detection
    results.append(test_nonzero_race_detection())
    
    # Test 2: Coercion cache threading
    results.append(test_coercion_cache_threading())
    
    # Test 3: Ufunc loop cache threading  
    results.append(test_ufunc_loop_cache_threading())
    
    print("\n" + "=" * 45)
    print("Summary:")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All threading safety tests passed!")
        return 0
    else:
        print("⚠ Some tests failed or had issues")
        return 1

if __name__ == "__main__":
    exit(main())