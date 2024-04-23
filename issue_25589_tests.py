import numpy as np
from numpy.ma.testutils import assert_equal

array1 = np.ma.array([3, 4, 5, 6], mask = [False, False, False, False])
result1_array1 = array1 ** 3.5
assert_equal(result1_array1.data, np.array([46.76537180435969, 128.0, 279.5084971874737, 529.0897844411664]))
assert_equal(len(result1_array1.mask), 4)

result2_array1 = array1 ** 0.9
assert_equal(result2_array1.data, np.array([2.6878753795222865, 3.4822022531844965, 4.256699612603923, 5.015752812467621]))
assert_equal(len(result2_array1.mask), 4)

result3_array1 = array1 ** 1.75
assert_equal(result3_array1.data, np.array([6.8385211708643325, 11.313708498984761, 16.71850762441055, 23.00195175286581]))
assert_equal(len(result3_array1.mask), 4)

array2 = np.ma.array([10, 10.7, 1.5, 19.3412], mask = [False])
result1_array2 = array2 ** 1.2391
assert_equal(result1_array2.data, np.array([17.342032668438673, 18.858599562101034, 1.6527024811874533, 39.271871725802804]))
assert_equal(len(result1_array2.mask), 4)

result2_array2 = array2 ** -0.6783
assert_equal(result2_array2.data, np.array([0.20974904879785425, 0.20034060622191774, 0.7595516280961806, 0.1340844142280547]))
assert_equal(len(result2_array2.mask), 4)

result3_array2 = array2 ** 5.1948
assert_equal(result3_array2.data, np.array([156602.9720686047, 222557.8146985335, 8.217862539806553, 4819744.856544941]))
assert_equal(len(result3_array2.mask), 4)

array3 = np.ma.array([3.4, 1.2], mask = [False, False])
result1_array3 = array3 ** 7.8

assert_equal(result1_array3.data, np.array([13980.91363494161, 4.145851281415363]))
assert_equal(len(result1_array3.mask), 2)

result2_array3 = array3 ** 1.2
assert_equal(result2_array3.data, np.array([4.342848711597634, 1.2445647472039776]))
assert_equal(len(result2_array3.mask), 2)

result3_array3 = np.ma.power(array3, 6.7)
assert_equal(result3_array3.data, np.array([3638.3857677420656, 3.3924569758842456]))
assert_equal(len(result3_array3.mask), 2)

array4 = np.ma.array([7.1205], mask = False)
result1_array4 = np.ma.power(array4, 4.57)
assert_equal(result1_array4, np.array([7869.967571527348]))
assert_equal(len(result1_array4.mask), 1)

result2_array4 = np.ma.power(array4, 0)
assert_equal(result2_array4, np.array([1]))
assert_equal(len(result2_array4.mask), 1)

array5 = np.ma.array([6.7], mask = [False])
result1_array5 = array5 ** 2.1
assert_equal(result1_array5, np.array([54.294655975192626]))
assert_equal(len(result1_array5.mask), 1)

result2_array5 = array5 ** -0.345
assert_equal(result2_array5, np.array([0.518805047913752]))
assert_equal(len(result2_array5.mask), 1)

result3_array5 = np.ma.power(array5, 1.24)
assert_equal(result3_array5, np.array([10.576275504399648]))
assert_equal(len(result3_array5.mask), 1)

print("All tests passed!")