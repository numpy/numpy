
import numpy as np
print("Features:")
print(f"Current: {np.core._multiarray_umath.__cpu_features__}")
print(f"Highway: {np.core._multiarray_umath.__hwy_features__}")

print("Baseline: ")
print(f"Current: {np.core._multiarray_umath.__cpu_baseline__}")
print(f"Highway: {np.core._multiarray_umath.__hwy_baseline__}")

print("Dispatch: ")
print(f"Current: {np.core._multiarray_umath.__cpu_dispatch__}")
print(f"Highway: {np.core._multiarray_umath.__hwy_dispatch__}")
