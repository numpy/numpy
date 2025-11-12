#!/usr/bin/env python3
"""
Basic test to verify same_value functionality works
"""

try:
    import numpy as np
    print("✓ NumPy imported successfully")
    
    # Test that same_value is accepted as a parameter
    try:
        result = np.can_cast(np.int32, np.int32, casting='same_value')
        print(f"✓ same_value parameter accepted, result: {result}")
    except Exception as e:
        print(f"✗ same_value parameter failed: {e}")
        exit(1)
    
    # Test a few basic cases
    print("\nTesting basic cases:")
    
    # Identity cast should work
    assert np.can_cast(np.int32, np.int32, casting='same_value')
    print("✓ Identity cast works")
    
    # Safe widening should work
    assert np.can_cast(np.int32, np.int64, casting='same_value')
    print("✓ Safe widening works")
    
    # Unsafe narrowing should fail
    assert not np.can_cast(np.int64, np.int32, casting='same_value')
    print("✓ Unsafe narrowing fails correctly")
    
    print("\n🎉 Basic same_value functionality is working!")
    
except ImportError:
    print("✗ Cannot import NumPy - this is expected if not in build environment")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    exit(1)
