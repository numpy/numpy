import numpy as np

# bug 25677
arr = np.ma.array([1, 2, 3], mask=[1, 0, 1], dtype=np.int8)
arr.filled()

arr = np.ma.array([1, 2, 3], mask=[1, 0, 1], dtype=np.float16)
arr.filled()

arr = np.ma.array([1.e4, 1.e4, 1.e4], mask = [1,1,1], dtype = np.float16)
arr.filled()

f16 = np.ma.array([230.12, 5601, 1027.8, 12389.4219], mask = [0, 0, 1, 0], dtype = "float16")
f16.filled()

f32 = np.ma.array([-230.12, 5601, 1027.8, 12389.4219, -9999.5321, 567.1034827], mask = [0, 0, 1, 0, 1, 1], dtype = "float32")
f32.filled()

f64 = np.ma.array([3.141592, -230.12, 5601, 1027.8, 12389.4219, -9999.5321, 567.1034827], mask = [0, 0, 1, 0, 1, 1, 1], dtype = "float64")
f64.filled()

i8 = np.ma.array([1, 2, 6, 7], mask = [1, 1, 0, 1], dtype = np.int8)
i8.filled()

ui8 = np.ma.array([1, 5], mask = [0, 1], dtype = np.uint8)
ui8.filled()

i16 = np.ma.array([5000, 120], mask = [1, 1], dtype = "int16")
i16.filled()

ui16 = np.ma.array([1, 5], mask = [0, 1], dtype = "uint16")
ui16.filled()

i32 = np.ma.array([11111, 5500, -9182], mask = [0, 1, 1], dtype = np.int32)
i32.filled()

ui32 = np.ma.array([11111, 5500, 9182, 4040], mask = [1, 0, 1, 1], dtype = np.uint32)
ui32.filled()

i64 = np.ma.array([pow(2, 34), pow(-2, 40), pow(2, 9)], mask = [0, 0, 0], dtype = np.int64)
i64.filled()

ui64 = np.ma.array([57, 120, pow(2, 34), pow(2, 40), pow(2, 41)], mask = [1, 1, 0, 0, 0], dtype = np.uint64)
ui64.filled()

i64 = np.ma.array([19675483920], mask=True, dtype = np.int64)
i64.filled()

a = np.ma.array([([0, 1],)], mask = np.ma.array([([False, False],)], dtype=[('A', '?', (2,))]), dtype=[('A', '>i2', (2,))])

ilist = [1, 2, 3, 4, 5]
flist = [1.1, 2.2, 3.3, 4.4, 5.5]
slist = [b'one', b'two', b'three', b'four', b'five']
ddtype = [('a', int), ('b', float), ('c', '|S8')]
mask = [0, 1, 0, 0, 1]
base = np.ma.array(list(zip(ilist, flist, slist)), mask=mask, dtype=ddtype)

# bug 26216
a = np.array([np.datetime64('2005-02-25'), np.datetime64(1, 'Y')])
print(np.median(a))

# bug 25589
a = np.array((1, 2, 3))  # arbitrary data
b = np.zeros(3)  # all-False mask
m = np.ma.masked_array(a, b)
m * 4
m ** 2
m ** 3
m ** 1.2
m ** -0.4

array1 = np.ma.array([3, 4, 5, 6], mask = [False, False, False, False])
array1 ** 3.5
array1 ** 0.9
array1 ** 1.75

array2 = np.ma.array([10, 10.7, 1.5, 19.3412], mask = [False])
array2 ** 1.2391
array2 ** -0.6783
array2 ** 5.1948

array3 = np.ma.array([3.4, 1.2], mask = [False, False])
array3 ** 7.8
array3 ** 1.2
np.ma.power(array3, 6.7)

array4 = np.ma.array([7.1205], mask = False)
np.ma.power(array4, 4.57)
np.ma.power(array4, 0)

array5 = np.ma.array([6.7], mask = [False])
array5 ** 2.1
array5 ** -0.345
np.ma.power(array5, 1.24)