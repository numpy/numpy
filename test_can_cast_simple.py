#!/usr/bin/env python3
"""
Simple test to verify same_value parameter in can_cast function
"""
import sys
import os

# Add the current numpy-work directory to the path
sys.path.insert(0, os.path.dirname(__file__))

try:
    import numpy as np
    print("NumPy version:", np.__version__)
    print("NumPy path:", np.__file__)
    
    # Test basic can_cast functionality first
    print("\n=== Testing basic can_cast functionality ===")
    print("can_cast('int32', 'float64', casting='safe'):", np.can_cast('int32', 'float64', casting='safe'))
    print("can_cast('float64', 'int32', casting='safe'):", np.can_cast('float64', 'int32', casting='safe'))
    
    # Test same_value parameter  
    print("\n=== Testing same_value casting parameter ===")
    try:
        result = np.can_cast('int32', 'int32', casting='same_value')
        print("SUCCESS: can_cast('int32', 'int32', casting='same_value'):", result)
    except Exception as e:
        print("FAILED: can_cast with same_value raised:", type(e).__name__, "-", e)
        
    try:
        result = np.can_cast('int32', 'int64', casting='same_value') 
        print("SUCCESS: can_cast('int32', 'int64', casting='same_value'):", result)
    except Exception as e:
        print("FAILED: can_cast with same_value raised:", type(e).__name__, "-", e)
        
except ImportError as e:
    print("FAILED: Cannot import numpy:", e)
except Exception as e:
    print("ERROR:", type(e).__name__, "-", e)