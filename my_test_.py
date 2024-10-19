import numpy as np

x = 0.123456789_123456789_123456789
array = np.ma.array([x,], dtype=np.float32, mask=False)
# array = np.array([x,], dtype=np.float32)

print(f"array : {type(array)} {array.dtype}", array)
print()

c = 1.0
a = (array * c)[0]
b = array[0] * c

print(f"a          : {a.dtype}", a)
print(f"b          : {b.dtype}", b)
print(f"difference : {(a-b).dtype}", a-b)
print()

c = 1e10
a = (array * c)[0]
b = array[0] * c

print(f"a          : {a.dtype}", a)
print(f"b          : {b.dtype}", b)
print(f"difference : {(a-b).dtype}", a-b)