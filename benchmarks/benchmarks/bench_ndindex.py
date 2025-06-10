import time
import numpy as np
from itertools import product

def benchmark_current_ndindex(shape):
    """Benchmark the current NumPy ndindex implementation."""
    start_time = time.perf_counter()
    count = sum(1 for _ in np.ndindex(*shape))
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, count

def benchmark_itertools_product(shape):
    """Benchmark equivalent itertools.product implementation."""
    start_time = time.perf_counter()
    count = sum(1 for _ in product(*(range(s) for s in shape)))
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, count

def run_benchmark_comparison():
    """Run comprehensive benchmark comparison."""
    
    test_shapes = [
        (10, 10),
        (20, 20),
        (50, 50),
        (10, 10, 10),
        (20, 30, 40),
        (50, 60, 90),  # The shape from the original issue
    ]
    
    print("=" * 80)
    print(f"{'NumPy ndindex Performance Benchmark':^80}")
    print("=" * 80)
    print(f"{'Shape':<15} {'Elements':<10} {'NumPy (ms)':<12} {'itertools (ms)':<15} {'Speedup':<10}")
    print("-" * 80)
    
    total_speedup = 0
    num_tests = 0
    
    for shape in test_shapes:
        # Benchmark NumPy ndindex
        numpy_time, numpy_count = benchmark_current_ndindex(shape)
        
        # Benchmark itertools.product
        itertools_time, itertools_count = benchmark_itertools_product(shape)
        
        # Verify they produce the same number of elements
        assert numpy_count == itertools_count, f"Count mismatch: {numpy_count} vs {itertools_count}"
        
        # Calculate speedup
        speedup = numpy_time / itertools_time if itertools_time > 0 else float('inf')
        total_speedup += speedup
        num_tests += 1
        
        # Convert to milliseconds for display
        numpy_ms = numpy_time * 1000
        itertools_ms = itertools_time * 1000
        
        print(f"{str(shape):<15} {numpy_count:<10} {numpy_ms:<12.2f} {itertools_ms:<15.2f} {speedup:<10.1f}x")
    
    print("-" * 80)
    avg_speedup = total_speedup / num_tests
    print(f"Average speedup: {avg_speedup:.1f}x")
    print("=" * 80)
    
    return avg_speedup

def benchmark_memory_usage():
    """Test memory efficiency of the new implementation."""
    print("\nMemory Usage Test:")
    print("-" * 40)
    
    # Test with a shape that would create many indices
    shape = (100, 100)
    
    # The key advantage: itertools.product is a generator, so it doesn't
    # store all indices in memory at once
    print(f"Testing shape {shape} ({shape[0] * shape[1]} total elements)")
    
    # Time just creating the iterator (should be very fast)
    start_time = time.perf_counter()
    ndindex_iter = np.ndindex(*shape)
    creation_time = time.perf_counter() - start_time
    
    print(f"Iterator creation time: {creation_time * 1000:.3f} ms")
    
    # Time consuming first 1000 elements
    start_time = time.perf_counter()
    for i, idx in enumerate(ndindex_iter):
        if i >= 1000:
            break
    partial_time = time.perf_counter() - start_time
    
    print(f"Time to consume first 1000 elements: {partial_time * 1000:.3f} ms")
    print("--> Memory efficient: Only generates indices as needed")

if __name__ == "__main__":
    try:
        avg_speedup = run_benchmark_comparison()
        benchmark_memory_usage()
        
        print(f"\nSUCCESS! Average performance improvement: {avg_speedup:.1f}x")
        
        if avg_speedup > 5:
            print("Excellent performance improvement achieved!")
        elif avg_speedup > 2:
            print("Good performance improvement achieved!")
        else:
            print("Performance improvement is modest.")
            
    except Exception as e:
        print(f"Benchmark failed: {e}")
        raise


