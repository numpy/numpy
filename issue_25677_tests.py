import numpy as np
from numpy.ma.testutils import assert_equal

float16_1 = np.ma.array([230.12, 5601, 1027.8, 12389.4219], mask = [0, 0, 1, 0], dtype = "float16")
assert_equal(float16_1.filled(), np.array([230.12, 5601, 65504, 12389.4219], dtype = np.float16))

float16_2 = np.ma.array([7.8, 1.204, -193], mask = [0, 0, 0], dtype = np.float16)
assert_equal(float16_2.filled(), np.array([7.8, 1.204, -193], dtype = np.float16))

float16_3 = np.ma.array([2951], mask = [1], dtype = np.float16)
assert_equal(float16_3.filled(), np.array([65504], dtype = np.float16))

float16_4 = np.ma.array([65504, 65504, 221.4612, 43391, -14583], mask = [1, 0, 1, 0, 1], dtype = np.float16)
assert_equal(float16_4.filled(), np.array([65504, 65504, 65504, 43391, 65504], dtype = np.float16))

int8_1 = np.ma.array([1, 2, 6, 7], mask = [1, 1, 0, 1], dtype = np.int8)
assert_equal(int8_1.filled(), np.array([127, 127, 6, 127], dtype = np.int8))

int8_2 = np.ma.array([127, 127], mask = [1, 0], dtype = np.int8)
assert_equal(int8_2.filled(), np.array([127, 127], dtype = np.int8))

int8_3 = np.ma.array([100], mask = [1], dtype = np.int8)
assert_equal(int8_3.filled(), np.array([127], dtype = np.int8))

int8_4 = np.ma.array([-14, 123, 104, -98], mask = [0, 0, 0, 1], dtype = np.int8)
assert_equal(int8_4.filled(), np.array([-14, 123, 104, 127], dtype = np.int8))

uint8_1 = np.ma.array([1, 5], mask = [0, 1], dtype = np.uint8)
assert_equal(uint8_1.filled(), np.array([1, 255], dtype = np.uint8))

uint8_2 = np.ma.array([19, 156], mask = [1, 1], dtype = np.uint8)
assert_equal(uint8_2.filled(), np.array([255, 255], dtype = np.uint8))

uint8_3 = np.ma.array([92, 57, 194, 1, 0, 12], mask = [1, 1, 1, 0, 1, 0], dtype = np.uint8)
assert_equal(uint8_3.filled(), np.array([255, 255, 255, 1, 255, 12], dtype = np.uint8))

uint8_4 = np.ma.array([1, 4, 12, 96, 45], mask = [0, 0, 0, 0, 1], dtype = np.uint8)
assert_equal(uint8_4.filled(), np.array([1, 4, 12, 96, 255], dtype = np.uint8))

int16_1 = np.ma.array([0, 120], mask = [1, 1], dtype = "int16")
assert_equal(int16_1.filled(), np.array([32767, 32767], dtype = np.int16))

int16_2 = np.ma.array([240], mask = [1], dtype = np.int16)
assert_equal(int16_2.filled(), np.array([32767], dtype = np.int16))

int16_3 = np.ma.array([-1, 42, 123, 50, 81, 217], mask = [0, 1, 0, 0, 0, 0], dtype = "int16")
assert_equal(int16_3.filled(), np.array([-1, 32767, 123, 50, 81, 217], dtype = np.int16))

int16_4 = np.ma.array([10, 128, -19, 1], mask = [0, 0, 1, 0], dtype = np.int16)
assert_equal(int16_4.filled(), np.array([10, 128, 32767, 1], dtype = np.int16))

uint16_1 = np.ma.array([1, 5], mask = [0, 1], dtype = "uint16")
assert_equal(uint16_1.filled(), np.array([1, 65535], dtype = np.uint16))

uint16_2 = np.ma.array([9, 79, 31, 111], mask = [1, 1, 1, 1], dtype = "uint16")
assert_equal(uint16_2.filled(), np.array([65535, 65535, 65535, 65535], dtype = np.uint16))

uint16_3 = np.ma.array([57, 12, 100], mask = [0, 0, 0], dtype = np.uint16)
assert_equal(uint16_3.filled(), np.array([57, 12, 100], dtype = np.uint16))

uint16_4 = np.ma.array([2, 56, 6, 1, 123, 61, 1], mask = [0, 0, 0, 1, 0, 1, 1], dtype = "uint16")
assert_equal(uint16_4.filled(), np.array([2, 56, 6, 65535, 123, 65535, 65535], dtype = np.uint16))

print("All tests passed!")