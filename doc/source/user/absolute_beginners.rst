****************************************
NumPy: The Absolute Basics for Beginners
****************************************

This is a working document for a future section introducing NumPy to absolute beginners. If you have comments or suggestions, please don’t hesitate to reach out!

Welcome to NumPy!
-----------------

NumPy (**Numerical Python**) is an open source Python library that's used in almost every field of science and engineering. It's the universal standard for working with numerical data in Python, and it's at the core of the scientific Python and PyData ecosystems. NumPy users include everyone from beginning coders to experienced researchers doing state-of-the-art scientific and industrial research and development. The NumPy API is used extensively in Pandas, SciPy, Matplotlib, scikit-learn, scikit-image and most other data science and scientific Python packages. 

The NumPy library contains multidimentional array and matrix data structures. It provides **ndarray**, a homogeneous n-dimensional array object with methods to efficiently operate on it. NumPy can be used to perform a wide variety of mathematical operations on arrays.  It enriches Python with powerful data structures that guarantee efficient calculations with arrays and matrices and it supplies an enormous library of high-level mathematical functions that operate on these array and matrices. 

`Learn more about NumPy here <https://docs.scipy.org/doc/numpy-1.17.0/user/whatisnumpy.html>`_!

Installing NumPy
----------------
  
To install NumPy, we strongly recommend using a scientific Python distribution. If you're looking for the full instructions for installing NumPy on your operating system, you can `find all of the details here <https://www.scipy.org/install.html>`_.

If you don't have Python yet, you might want to consider using Anaconda. It's the easiest way to get started! The good thing about getting this distribution is is the fact that you don’t need to worry too much about separately installing NumPy or any of the major packages that you’ll be using for your data analyses, such as pandas, Scikit-Learn, etc.
  
If you already have Python, you can install NumPy with

::

  conda install numpy
  
or 

::

  pip install numpy
  
You can find all of the installation details in the `Installation <https://www.scipy.org/install.html>`_ section at scipy.org.

How to import NumPy
-------------------

Any time you want to use a package or library in your code, you first need to make it accessible. 

In order to start using NumPy and all of the functions available in NumPy, you'll need to import it. This can be easily done with this import statement:

::

  import numpy as np 

(We shorten "numpy" to "np" in order to save time and also to keep code standardized so that anyone working with your code can easily understand and run it.)

What’s the difference between a Python List and a NumPy array? 
--------------------------------------------------------------
  
NumPy gives you an enormous range of fast and efficient numerically-related options. While a Python list can contain different data types within a single list, all of the elements in a NumPy array should be homogenous. The mathematical operations that are meant to be performed on arrays wouldn't be possible if the arrays weren't homogenous. 

**Why use NumPy?**

NumPy arrays are faster and more compact than Python lists. An array consumes less memory and is convenient to use. NumPy uses much less memory to store data and it provides a mechanism of specifying the data types. This allows the code to be optimised even further. 

What is an array?
-----------------

An array is a central data structure of the NumPy library. It's a grid of values and it contains information about the raw data, how to locate an element, and how to interpret an element. It has a grid of elements that can be indexed in `various ways <https://numpy.org/devdocs/user/quickstart.html#indexing-slicing-and-iterating>`_. The elements are all of the same type, referred to as the array **dtype**. 

An array can be indexed by a tuple of nonnegative integers, by booleans, by another array, or by integers. The **rank** of the array is the number of dimensions. The **shape** of the array is a tuple of integers giving the size of the array along each dimension.

One way we can initialize NumPy arrays is from nested Python lists. 

::

  a = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

We can access the elements in the array using square brackets. When you're accessing elements, remember that indexing in NumPy starts at 0. That means that, if you want to access the first element in your array, you'll be accessing element "0".

::

  print(data[0])

More information about arrays
-----------------------------

**What else might an array be called?**

You might occasionally hear an array referred to as an "ndarray," which is shorthand for "N-dimensional array." You might also hear **1-D**, or one-dimensional array, **2-D**, or two-dimensional array, and so on. The numpy `ndarray` class is used to represent both matrices and vectors. A vector is an array with a single column, while a matrix refers to an array with multiple columns.

**What are the attributes of an array?**

An array is usually a fixed-size container of items of the same type and size. The number of dimensions and items in an array is defined by its shape. The shape of an array is a tuple of non-negative integers that specify the sizes of each dimension. 

In NumPy, dimensions are called **axes**. This means that if you have a 2D array that looks like this:

::

  [[0., 0., 0.],
   [1., 1., 1.]]

Your array has 2 axes. The first axis has a length of 2 and the second axis has a length of 3.

Just like in other Python container objects, the contents of an array can be accessed and modified by indexing or slicing the array. Different arrays can share the same data, so changes made on one array might be visible in another. 

Array **attributes** reflect information intrinsic to the array itself. If you need to get, or even set, poperties of an array without creating a new array, you can often access an array through its attributes. 

`Read more about array attributes here <https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html>`_ and learn about `array objects here <https://docs.scipy.org/doc/numpy-1.17.0/reference/arrays.html>`_.


How to create a basic array
---------------------------

To create a NumPy array, you can use the function `np.array()`

All you need to do to create a simple array is pass a list to it. If you choose to, you can also specify the type of data in your list. `You can find more information about data types here <https://numpy.org/devdocs/user/quickstart.html#arrays-dtypes>`_.

::

    import numpy as np

    # create a 1-D array
    a = np.array([1, 2, 3])

The default data type is floating point and you can explicity specify which data type you want any time:

::

  b = np.array([1, 2, 3], dtype=float)

You can visualize your array this way:

.. image:: images/np_array.png

Besides creating an array from a sequence of elements, you can easily create an array filled with 0s:

::

  # Create a 1D array with 2 elements, both 0s
  np.zeros(2)

**Output:**

::

  array([0., 0.])

Or an array filled with 1s:

::

  # Create a 1D array with 2 eleements, both 1s
  np.ones(2)

**Output:**

::

  array([1., 1.])
  
Or even an empty array! The function *empty* creates an array whose initial content is random and depends on the state of the memory. 

::

  # Create an empty array with 2 elements
  np.empty(2)

You can create an array with a range of elements:

::

  # Create a 1D array containing the numbers 0,1,2,3
  np.arange(4)

**Output:**

::

  array([0, 1, 2, 3])

And even an array that contains a range of evenly spaced interval. To do this, you will specify the first and last number and the step size.

::

  np.arange(2,9,2)

**Output:**

::

  array([2, 4, 6, 8])

It's simple to create an array where the values are spaced linearly in an interval:

::

  np.linspace(0,10,5)

**Output:**

::

  array([ 0. ,  2.5,  5. ,  7.5, 10. ])

While the default data type is floating point (float64), you can expecity specify which data type you want using 'dtype'.

::

  array = np.ones(2, dtype=int)
  array

**Output:**

::

  array([1, 1])

`Learn more about creating arrays here <https://docs.scipy.org/doc/numpy-1.17.0/user/quickstart.html#array-creation>`_.

Adding, removing, and sorting elements
--------------------------------------

Let's take advantage of:

::

  np.append()
  np.delete()
  np.sort()

If we start with this array:

::

  arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
 

**Append**

You can add elements to an array any time with np.append.
::

  np.append(arr, [1,2])

**Output**

::

  array([1, 2, 3, 4, 5, 6, 7, 8, 1, 2])

**Delete**

You can delete an element with np.delete. 

::

  # Delete the element in position 1
  np.delete(arr, 1)

**Output**

::

  array([1, 3, 4, 5, 6, 7, 8])

**Sort**

Sorting an element is simple with np.sort. You can specify the axis, kind, and order when you call the function. `Read more about sorting an array here <https://docs.scipy.org/doc/numpy/reference/generated/numpy.sort.html>`_.

If you start with this array:

::

  arr2 = np.array([2, 1, 5, 3, 7, 4, 6, 8])

You can quickly sort the numbers in ascending order with:

::

  np.sort(arr2)

**Output:**

::

  array([1, 2, 3, 4, 5, 6, 7, 8])

In addition to sort, which returns a sorted copy of an array, you can use:

**argsort**, which is an `indirect sort along a specified axis <https://docs.scipy.org/doc/numpy-1.17.0/reference/generated/numpy.argsort.html#numpy.argsort>`_,
**lexsort**, which is an `indirect stable sort on multiple keys <https://docs.scipy.org/doc/numpy-1.17.0/reference/generated/numpy.lexsort.html#numpy.lexsort>`_,
**searchsorted**, which will `find elements in a sorted array <https://docs.scipy.org/doc/numpy-1.17.0/reference/generated/numpy.searchsorted.html#numpy.searchsorted>`_, and 
**partition**, which is a `partial sort  <https://docs.scipy.org/doc/numpy-1.17.0/reference/generated/numpy.partition.html#numpy.partition>`_.


How do you know the shape and size of an array?
-----------------------------------------------

**ndarray.ndim** will tell you the number of axes, or dimensions, of the array.

**ndarray.size** will tell you the total number of elements of the array. This is the *product* of the elements of the array's shape.

**ndarray.shape** will display a tuple of integers that indicate the number of elements stored along each dimension of the array. If, for example, you have a 2D array with 2 rows and 3 columns, the shape of your array is (2,3).

For example:

::

      import numpy as np
      array_example = np.array([[[0, 1, 2, 3]
                                 [4, 5, 6, 7]],

                                 [[0, 1, 2, 3]
                                  [4, 5, 6, 7]],

                                  [0 ,1 ,2, 3]
                                  [4, 5, 6, 7]]])

  array_example.ndim # Number of dimensions
  array_example.size # Total number of elements in the array
  array_example.shape # Shape of your array

**Output:**

::

  3
  24
  (3,2,4)


Can you reshape an array?
-------------------------
  
**Yes!**

::

  numpy.reshape() 

will give a new shape to an array without changing the data. Just remember that when you use the reshape method, the array you want to produce needs to have the same number of elements as the original array. If you start with an array with 12 elements, you'll need to make sure that your new array also has a total of 12 elements.

For example:

::

  a = np.arange(6)
  print('Original array:')
  print(a)
  print('\n')

  b = a.reshape(3,2)
  print('Modified array:')
  print(b)

**Output:**

::

  Original array:
  [0 1 2 3 4 5]

  Modified array:
  [[0 1]
   [2 3]
   [4 5]]

You can specify a few optional parameters.

::

  numpy.reshape(a, newshape, order)

**a** is the array to be reshaped.

**newshape** is the new shape you want. You can specify an integer or a tuple of integers. If you specify an integer, the result wil be an array of that length. The shape should be compatible with the original shape.

**order:** 'C' means to read/write the elements using C-like index order,  ‘F’ means to read / write the elements using Fortran-like index order, ‘A’ means to read / write the elements in Fortran-like index order if a is Fortran contiguous in memory, C-like order otherwise. (This is an optional parameter and doesn't need to be specified.)

`Learn more about shape manipulation here <https://docs.scipy.org/doc/numpy-1.17.0/user/quickstart.html#shape-manipulation>`_.

Indexing and Slicing
--------------------

We can index and slice NumPy arrays in the same ways we can slice Python lists.

::

   # create a 1-D array
    data = np.array([1,2,3])

    # print the first element of the array
    print(data[0])
    print(data[1])
    print(data[0:2])
    print(data[1:])
    print(data[-2:])

**Output:**

::

  1
  2
  [1 2]
  [2 3]
  [2 3]

You can visualize it this way:

.. image:: images/np_indexing.png

`Learn more about indexing and slicing here <https://docs.scipy.org/doc/numpy-1.17.0/user/quickstart.html#indexing-slicing-and-iterating>`_ and `here <https://docs.scipy.org/doc/numpy-1.17.0/user/basics.indexing.html>`_.

How to create an array from existing data
-----------------------------------------

You can easily create a new array from a section of an existing array. Let's say you have this array:

::

  array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

You can create a new array from a section of your array any time by specifying where you want to slice your array.

::

  arr1 = arr[3:8]
  arr1

**Output:**

::

  array([4, 5, 6, 7, 8])

Here, you grabbed a section of your array from index position 3 through index position 8.

You can also stack two existing arrays, both vertically and horizontally. Let's say you have two arrays. This one:

::

  array([[1, 1],
       [2, 2]])

and this one:

::

  array([[3, 3],
       [4, 4]])

You can stack them vertically with vstack:

::

  np.vstack((a_1, a_2))

**Output:**

::

  array([[1, 1],
       [2, 2],
       [3, 3],
       [4, 4]])

Or stack them horizontally with hstack:

::

  np.hstack((a_1, a_2))

**Output:**

::

  array([[1, 1, 3, 3],
       [2, 2, 4, 4]])

`Learn more about stacking and splitting arrays here <https://docs.scipy.org/doc/numpy-1.17.0/user/quickstart.html#stacking-together-different-arrays>`_.

You can also split an array into several smaller arrays using hsplit. You can specify either the number of equally shaped arrays to return or the columns *after* which the division should occur.

Let's say you have this array:

::

  array([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
       [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]])

If you wanted to split this array into three equally shaped arrays, you would run:

::

  np.hsplit(a_3,3)

**Output:**

::

  [array([[ 1,  2,  3,  4],
        [13, 14, 15, 16]]), array([[ 5,  6,  7,  8],
        [17, 18, 19, 20]]), array([[ 9, 10, 11, 12],
        [21, 22, 23, 24]])]

If you wanted to split your array after the third and fourth column, you'd run:

::

  np.hsplit(a_3,(3,4))

**Output:**

::

  [array([[ 1,  2,  3],
        [13, 14, 15]]), array([[ 4],
        [16]]), array([[ 5,  6,  7,  8,  9, 10, 11, 12],
        [17, 18, 19, 20, 21, 22, 23, 24]])]

You can also use the `view` method to create a new array object that looks at the same data (a *shallow copy*)

Let's say you create this array:

::

  a = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

You can create a new array with the same data using:

::

  b = np_arr.view()

Using the `copy` method makes a complete copy of the array and its data (a *deep copy*). To use this on your array, you could run:

::

  c = a.copy()
 
`Learn more about copies and views here <https://docs.scipy.org/doc/numpy-1.17.0/user/quickstart.html#copies-and-views>`_.

Basic array operations
----------------------

Once you've created your arrays, you can start to work with them. Let's say, for example, that you've created two arrays, one called "data" and one called "ones" 

.. image:: images/np_array_dataones.png

You can easily add the arrays together with the plus sign.

::

  data + ones

.. image:: images/np_data_plus_ones.png

Of course, you can do more than just addition!

::

  data - ones
  data * data
  data / data

.. image:: images/np_sub_mult_divide.png

Basic operations are simple with NumPy. If you want to find the sum of the elements in an array, you'd use sum(). This works for 1D arrays, 2D arrays, and arrays in higher dimentions.

::

  a = np.array([1, 2, 3, 4])

  # Add all of the elements in the array
  a.sum()

**Output:**

::

  10

To add the rows or the columns in a 2D array, you would specify the axis.

::

  b = np.array([[1, 1], [2, 2]])

  # Sum the rows
  b.sum(axis=0)

**Output:**

::

  array([3, 3])

::

  # Sum the columns
  b.sum(axis=1)

**Output:**

::

  array([2, 4])

`Learn more about basic operations here <https://docs.scipy.org/doc/numpy-1.17.0/user/quickstart.html#basic-operations>`_.


Broadcasting
------------

There are times when you might want to carry out an operation between an array and a single number (also called *an operation between a vector and a scalar*). Your array (we'll call it "data") might, for example, contain information about distance in miles but you want to convert the information to kilometers. You can perform this operation with: 

::

  data * 1.6

.. image:: images/np_multiply_broadcasting.png

NumPy understands that the multiplication should happen with each cell. That concept is called **broadcasting**.

`Learn more about broadcasting here <https://docs.scipy.org/doc/numpy-1.17.0/user/basics.broadcasting.html>`_.


More useful array operations
-----------------------------------

NumPy also performs aggregation functions. In addition to `min`,  `max`, and `sum`, you can easily run `mean` to get the average, `prod` to get the result of multiplying the elements together, `std` to get the standard deviation, and more.

::

  data.max()
  data.min()
  data.sum()

.. image:: images/np_aggregation.png

Let's start with this array, called "A"

::

 [[0.45053314 0.17296777 0.34376245 0.5510652]
 [0.54627315 0.05093587 0.40067661 0.55645993]
 [0.12697628 0.82485143 0.26590556 0.56917101]]

It's very common to want to aggregate along a row or column. By default, every NumPy aggregation function will return the aggregate of the entire array. To find the sum or and the minimum of the elements in your array, simply run:

::

  A.sum()

Or

::

  A.min()

**Output:**

::

  # Sum
  4.8595783866706

  # Minimum
  0.050935870838424435

You can easily specify which axis you want the aggregation function to be computed. For example, you can find the minimum value within each column by specifying `axis=0`.

::

  A.min(axis=0)

**Output:**

::

  array([0.12697628, 0.05093587, 0.26590556, 0.5510652 ])

The four values listed above correspond to the number of columns in your array. With a four-column array, you can expect to get four values as your result.

`Read more about functions here <https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html>`_ and `calculations here <https://docs.scipy.org/doc/numpy-1.17.0/reference/arrays.ndarray.html#calculation>`_.


How to inspect the size and shape of a NumPy array
--------------------------------------------------

You can get the dimensions of a NumPy array any time using ndarray.shape and NumPy will return the dimensions of the array as a tuple.

For example, if you created this array:

::

  np_arr = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
 
  print(np_arr)

**Output:**

::

  [[ 1  2  3  4]
  [ 5  6  7  8]
  [ 9 10 11 12]]

You can use `.shape` to quickly find the shape of your array:

::

  np_arr.shape

**Output:**

::

  (3, 4)

This output tells you that your array has three rows and four columns.

You can find just the number of rows by specifying [0]:

::

  num_of_rows = np_arr.shape[0]
 
  print('Number of Rows : ', num_of_rows)

**Output:**

::

  Number of Rows :  3

Or just the number of columns by specifying [1]:

::

  num_of_columns = np_arr.shape[1]
 
  print('Number of Columns : ', num_of_columns) 

**Output:**

::
  
  Number of Columns :  4

It's also easy to find the total number of elements in your array:

::

  # np_arr.shape[0] * np_arr.shape[1]

  print('Total number of elements in array : ', np_arr.shape[0] * np_arr.shape[1])

**Output:**

::

  Total number of elements in array:  12

You can use np.shape() with a 1D array, of course.

::

  # Create an array
  arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

  print('Shape of 1D array: ', arr.shape)
  print('Length of 1D array: ', arr.shape[0])

**Output:**

::

  Shape of 1D array:  (8,)
  Length of 1D array:  8


You can get the dimensions of an array using np.size()

::

  # get number of rows in array
  num_of_rows2 = np.size(np_arr, 0)
 
  # get number of columns in 2D numpy array
  num_of_columns2 = np.size(np_arr, 1)
 
  print('Number of Rows : ', num_of_rows2)
  print('Number of Columns : ', num_of_columns2)

**Output:**

::

  Number of Rows :  3
  Number of Columns: 4

You can print the total number of elements as well:

::
  
  print('Total number of elements in  array : ', np.size(np_arr))

**Output:**

::

  Total number of elements in  array :  12

This also works for 3D arrays:

::

  arr3D = np.array([ [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
                 [[4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]] ])
 
  print(arr3D)

**Output:**

::

  [[[1 1 1 1]
    [2 2 2 2]
    [3 3 3 3]]

  [[4 4 4 4]
    [5 5 5 5]
    [6 6 6 6]]]

You can easily print the size of the axis:

::

  print('Axis 0 size : ', np.size(arr3D, 0))
  print('Axis 1 size : ', np.size(arr3D, 1))
  print('Axis 2 size : ', np.size(arr3D, 2))

**Output:**

::

  Axis 0 size :  2
  Axis 1 size :  3
  Axis 2 size :  4

You can print the total number of elements:

::

  print('Total number of elements in 3D Numpy array : ', np.size(arr3D))

**Output:**

::

  Total number of elements in 3D Numpy array :  24

You can also use np.size() with 1D arrays:

::

  # Create a 1D array
  arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

  # Determine the length
  print('Length of 1D numpy array : ', np.size(arr))

**Output:**

::

  Length of 1D numpy array :  8

Remember that if you check the size of your array and it equals 0, your array is empty.


Creating Matrices
-----------------

You can pass Python lists of lists to create a matrix to represent them in NumPy.

::

  np.array([[1,2],[3,4]])

.. image:: images/np_create_matrix.png

Indexing and slicing operations can be useful when you're manipulating matrices:

::

  data[0,1]
  data[1:3]
  data[0:2,0]

.. image:: images/np_matrix_indexing.png

You can aggregate matrices the same way you aggregated vectors:

::

  data.max()
  data.min()
  data.sum()

.. image:: images/np_matrix_aggregation.png

You can aggregate all the values in a matrix and you can aggregate them across columns or rows using the `axis` parameter:

::
  
  data.max(axis=0)
  data.max(axis=1)


.. image:: images/np_matrix_aggregation_row.png

Once you've created your matrices, you can add and multiply them using arithmetic operators if you have two matrices that are the same size.

::

  data + ones

.. image:: images/np_matrix_arithmetic.png

You can do these arithmetic operations on matrices of different sizes, but only if the different matrix has only one column or onw row. In this case, NumPy will use its broadcast rules for the operation.

::

  data + ones_row

.. image:: images/np_matrix_broadcasting.png

- How to extract specific items from an array
- How to create sequences, repetitions, and random numbers

NumPy can do everything we've mentioned in any number of dimensions, that's why it's called an N-Dimensional array.

Be aware that when NumPy prints N-Dimensional arrays, the last axis is looped over the fastest while the first axis is the slowest. That means that 

::

  np.ones((4,3,2))

Will print out like this:

**Output:**

::

  array([[[1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1., 1.],
        [1., 1.],
        [1., 1.]]])

 
There are often instances where we want NumPy to initialize the values of an array. NumPy offers methods like ones(), zeros() and random.random() for these instances. All you need to do is pass in the number of elements you want it to generate.

::

  np.ones(3)
  mp.zeros(3)
  np.random.random((3)
  
.. image:: images/np_ones_zeros_random.png

Generating random numbers
-------------------------

The use of random number generatiion is an important part of the configuration and evaluation of machine learning algorithms. Whether you neeed to randomly initialize weights in an artificial neural network, split data into random sets, or randomly shuffle your dataset, being able to generate random numbers (actually, repeatable pseudo-random numbers) is essential.

You have a number of options when using NumPy for random number generation. Random Generator is NumPy's replacement for RandomState. The main difference between them is that Generator relies on an additional BitGenerator to manage state and generate the random bits, which are transformed into random values.

With Generator.integers, you can generate random integers from low (remeber that this is inclusive with NumPy) to high (exclusive). You can set *endopoint=True* to make the high number inclusive. 

You can generate a 2 x 4 array of random integers between 0 and 4 with

::

  rng.integers(5, size=(2, 4))

**Output:**

::

  array([[4, 0, 2, 1],
       [3, 2, 2, 0]])


You can also use the `ones()`, `zeros()`, and `random()` methods to create an array if you give them a tuple describing the deminsions of the matrix.

::

  np.ones(3,2)
  mp.zeros(3,2)
  np.random.random((3,2)

.. image:: images/np_ones_zeros_matrix.png


How to get the unique items and the counts
------------------------------------------

How to get index locations that satisfy a given condition 
---------------------------------------------------------

Transposing and reshaping a matrix
----------------------------------

It's common to need to rotate your matrices. NumPy arrays have the property `T` that allows you to transpose a matrix.

.. image:: images/np_transposing_reshaping.png

You may need to switch the dimensions of a matrix. This can happen when, for example you have a model that expects a certain input shape that might be different from your dataset. This is where the `reshape` method can be useful. You pass in the new dimensions that you want for the matrix.

::

  data.reshape(2,3)
  data.reshape(3,2)

.. image:: images/np_reshape.png

How to reverse
--------------
 
NumPy's np.flip() function allows you to easily flip the contents of an array along an axis. You simply specify the array you would like to reverse and the axis. If you don't specify the axis, NumPy will flip or reverse the contents along all of the axes of your input array. 

**Reversing a 1D array**

If you begin with a 1D array like this one:

::

  arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

You can reverse it with: 

::

  reversedArr = np.flip(arr)

If you want to print your reversed array, you could run:

::

  print('Reversed Array: ', reversedArr)

**Output:**

::

  Reversed Array:  [8 7 6 5 4 3 2 1]

**Reversing a 2D array**

A 2D array works much the same way.

If you start with this array:

::

  arr2D = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

You can reverse the content in all of the rows and all of the columns with:

::

  reversedArr = np.flip(arr2D)
 
  print('Reversed Array: ')
  print(reversedArr)

**Output:**

::

  Reversed Array: 
  [[12 11 10  9]
   [ 8  7  6  5]
   [ 4  3  2  1]]

You can easily reverse only the rows with:

::

  reversedArr_rows = np.flip(arr2D, axis=0)
 
  print('Reversed Array: ')
  print(reversedArr_rows)

**Output:**

::

  Reversed Array: 
  [[ 9 10 11 12]
   [ 5  6  7  8]
   [ 1  2  3  4]]

Or reverse only the columns with:

::

  reversedArr_columns = np.flip(arr2D, axis=1)
 
  print('Reversed Array columns: ')
  print(reversedArr_columns)

**Output:**

::

  Reversed Array columns: 
  [[ 4  3  2  1]
   [ 8  7  6  5]
   [12 11 10  9]]

You can also reverse the contents of only one column or row. For example, you can reverse the contents of the row at index position 1 (the second row):

::

  arr2D[1] = np.flip(arr2D[1])
   
  print('Reversed Array: ')
  print(arr2D)

**Output:**

::

  Reversed Array: 
  [[ 1  2  3  4]
   [ 5  6  7  8]
   [ 9 10 11 12]]

You can also reverse the column at index position 1 (the second column):

::

  arr2D[:,1] = np.flip(arr2D[:,1])
   
  print('Reversed Array: ')
  print(arr2D)

**Output:**

::

  Reversed Array: 
  [[ 1 10  3  4]
   [ 5  6  7  8]
   [ 9  2 11 12]]


Reshaping and Flattening multidimensional arrays
------------------------------------------------
  
There are two popular ways to flatten an array: **flatten()** and **ravel()**. The primary difference between the two is that the new array created using **ravel()** is actually a reference to the parent array. This means that any changes to the new array will affect the parent array as well. Since ravel does not create a copy, it's memory efficient. 

If you start with this array:

::

  array = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

You can use **flatten()** to flatten your array into a 1D array.

::

  array.flatten()

**Output:**

::

  array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

When you use **flatten()**, changes to your new array won't change the parent array.

For example:

::

  a1 = array.flatten()  
  a1[0] = 100
  print('Original array: ')
  print(array)
  print('New array: ')
  print(a1)

**Output:**

::

  Original array: 
  [[ 1  2  3  4]
   [ 5  6  7  8]
   [ 9 10 11 12]]
  New array: 
  [100   2   3   4   5   6   7   8   9  10  11  12]


But when you use **ravel()**, the changes you make to the new array will affect the parent array.

For example:

::

  a2 = array.ravel()  
  a2[0] = 101 
  print('Original array: ')
  print(array)
  print('New array: ')
  print(a2)

**Output:**

::

  Original array: 
  [[101   2   3   4]
   [  5   6   7   8]
   [  9  10  11  12]]
  New array: 
  [101   2   3   4   5   6   7   8   9  10  11  12]


How to save and load NumPy objects
----------------------------------


How to convert a 1D array into a 2D array (how to add a new axis)
-----------------------------------------------------------------

Formulas
---------

Implementing mathematical formulas that work on matrices and vectors is one of the things that make NumPy so highly regarded in the scientific Python community. 

For example, this is the mean square error formula (a central formula used in supervised machine learning models that deal with regression):

.. image:: images/np_MSE_formula.png

Implementing this formula is simple and straightforward in NumPy:

.. image:: images/np_MSE_implementation.png

What makes this work so well is that `predictions` and `labels` can contain one or a thousand values. They only need to be the same size. 

You can visualize it this way:

.. image:: images/np_mse_viz1.png

In this example, both the predictions and labels vectors contain three values, meaning `n` has a value of three. After we carry out subtractions the values in the vector are squared. Then NumPy sums the values, and your result is the error value for that prediction and a score for the quality of the model.

.. image:: images/np_mse_viz2.png

.. image:: images/np_MSE_explanation2.png



Importing and exporting a CSV
-----------------------------

It's simple to read in a CSV that contains existing information. The best and easiest way to do this is to use Pandas.

::

  import pandas as pd

  # If all of your columns are the same type:
  x = pd.read_csv('music.csv').values

  # You can also simply select the columns you need:
  x = pd.read_csv('music.csv', columns=['float_colname_1', ...]).values

.. image:: images/np_pandas.png

It's simple to use Pandas in order to export your array as well. If you are new to NumPy, you may want to  create a pandas dataframe from the values in your array and then write the data frame to a CSV file with pandas.

If you created this array "a"

::

  [[-2.58289208,  0.43014843, -1.24082018,  1.59572603],
  [ 0.99027828,  1.17150989,  0.94125714, -0.14692469],
  [ 0.76989341,  0.81299683, -0.95068423,  0.11769564],
  [ 0.20484034,  0.34784527,  1.96979195,  0.51992837]]

You could create a Pandas dataframe

::

  df = pd.DataFrame(a)
  print(df)

.. image:: images/np_pddf.png

You can easily save your dataframe with

::

  df.to_csv('pd.csv')

And read your CSV with

::

  pd.read_csv('pd.csv')

.. image:: images/np_readcsv.png

You can also save your array with the NumPy "savetxt" method.

::

  np.savetxt('np.csv', a, fmt='%.2f', delimiter=',', header=" 1,  2,  3,  4")

Read your saved CSV any time with a command such as

::

  cat np.csv

**Output:**

::

  #  1,  2,  3,  4
  -2.58,0.43,-1.24,1.60
  0.99,1.17,0.94,-0.15
  0.77,0.81,-0.95,0.12
  0.20,0.35,1.97,0.52


Plotting arrays with Matplotlib
-------------------------------

If you need to generate a plot for your values, it's very simple with Matplotlib. 

For example, you may have an array like this one:

::

  A = np.array([2, 1, 5, 7, 4, 6, 8, 14, 10, 9, 18, 20, 22])

If you already have Matplotlib installed, you can import it with

::
  
  import matplotlib.pyplot as plt
  # If you're using Jupyter Notebook, you may also want to run the following line of code
   to display your code in the notebook
  %matplotlib inline

All you need to do to plot your values is run

::

  plt.plot(A)
  plt.show()

**Output:**

.. image:: images/np_matplotlib.png

For example, you can plot a 1D array like this:

::

  x = np.linspace(0, 5, 20)
  y = np.linspace(0, 10, 20)
  plt.plot(x, y, 'purple') # line  
  plt.plot(x, y, 'o')      # dots

.. image:: images/np_matplotlib1.png
    :scale: 50 %

With Matplotlib, you have access to an enormous number of visualization options.

::

  image = np.random.rand(40, 40)
  plt.imshow(image, cmap=plt.cm.magma)

  plt.colorbar()

.. image:: images/np_matplotlib2.png
    :scale: 50 %

To read more about Matplotlib and what it can do, take a look at `the official documentation <https://matplotlib.org/>`_.


How to read a docstring with `?` and source code with `??` in IPython/Jupyter
-----------------------------------------------------------------------------

More useful functions
---------------------

- np.clip

- np.digitize

- np.bincount

- np.histogram





-------------------------------------------------------

*Image credits: Jay Alammar http://jalammar.github.io/*

