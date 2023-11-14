import numpy as np

def helper(t1, t2):
    """
    Helper function for check_broadcast method.

    Parameters:
    - t1 (tuple): The first tuple.
    - t2 (tuple): The second tuple.

    Returns:
    - bool: True if the tuples can be broadcasted together, False otherwise.
    """
    t1_list = list(t1)
    t2_list = list(t2)

    ones_t1 = [i for i, value in enumerate(t1_list) if value == 1]
    ones_t2 = [i for i, value in enumerate(t2_list) if value == 1]

    for i in ones_t1:
        t1_list[i] = t2_list[i]

    for i in ones_t2:
        t2_list[i] = t1_list[i]

    return t1_list == t2_list


def check_broadcast(a, b):
    """
    Checks whether two NumPy arrays can be broadcasted together.

    Parameters:
    - a (numpy.ndarray): The first array.
    - b (numpy.ndarray): The second array.

    Returns:
    - bool: True if the arrays can be broadcasted together, False otherwise.
    """
    # If the number of dimensions of the first array is less than the number
    # of dimensions of the second array, swap the arrays so that the first
    # array has more dimensions.
    if a.ndim < b.ndim:
        a, b = b, a

    # Create a temporary copy of the shape of the second array.
    temp_shape = list(b.shape)

    # While the length of the temporary shape is not equal to the length of
    # the shape of the first array, insert ones into the temporary shape.
    while len(temp_shape) != len(a.shape):
        temp_shape.insert(0, 1)

    # Reshape the second array to have the same shape as the temporary shape.
    b = b.reshape(tuple(temp_shape))

    # If the number of dimensions of the first array is equal to the number
    # of dimensions of the second array, check if either array has a
    # dimension of 1.
    if a.ndim == b.ndim:
        if 1 in a.shape or 1 in b.shape:
            return helper(a.shape, b.shape)

    return False


## testing the function

# Test cases for arrays that can be broadcasted
array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[1], [2]])
print(check_broadcast(array1, array2))  # Should return True

array3 = np.array([1, 2, 3])
array4 = np.array([[1], [2], [3]])
print(check_broadcast(array3, array4))  # Should return True

array5 = np.array([1, 2, 3])
array6 = np.array([[1]])
print(check_broadcast(array5, array6))  # Should return True

array7 = np.array([1, 2, 3])
array8 = np.array([[1, 2, 3]])
print(check_broadcast(array7, array8))  # Should return True

array9 = np.array([1])
array10 = np.array([[1]])
print(check_broadcast(array9, array10))  # Should return True

# Test cases for arrays that cannot be broadcasted
array11 = np.array([1, 2, 3])
array12 = np.array([4, 5])
print(check_broadcast(array11, array12))  # Should return False

array13 = np.array([[1, 2], [3, 4]])
array14 = np.array([[1, 2, 3], [4, 5, 6]])
print(check_broadcast(array13, array14))  # Should return False

array15 = np.array([[1, 2, 3], [4, 5, 6]])
array16 = np.array([[1, 2]])
print(check_broadcast(array15, array16))  # Should return False

# Test cases with arrays of different dimensions
array17 = np.array([[1, 2, 3], [4, 5, 6]])
array18 = np.array([1, 2, 3])
print(check_broadcast(array17, array18))  # Should return True

array19 = np.array([[1, 2, 3], [4, 5, 6]])
array20 = np.array([[1], [2]])
print(check_broadcast(array19, array20))  # Should return True

array21 = np.array([[1, 2, 3], [4, 5, 6]])
array22 = np.array([[1, 2]])
print(check_broadcast(array21, array22))  # Should return False
