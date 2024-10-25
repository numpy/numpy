import numpy as np

k_real = np.array([1.,2.])
k_complex = k_real + 1j * k_real
k_real_masked = np.ma.MaskedArray(k_real, mask=False, fill_value=1e20)
k_complex_masked = np.ma.MaskedArray(k_complex, mask=False, fill_value=1e20)

print("###")
print(np.sqrt(k_real))
print(np.sqrt(k_complex))
print(np.sqrt(k_real_masked))
print(np.sqrt(k_complex_masked))

print("###")
print(np.sqrt(1j * k_real))
print(np.sqrt(1j * k_complex))
print(np.sqrt(1j * k_real_masked))
print(np.sqrt(1j * k_complex_masked))

print("###")
print(np.sqrt(1j * k_complex_masked.data))
print(np.sqrt(1j * k_complex_masked).data)