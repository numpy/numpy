**************
NumPy: The Absolute Basics for Beginners
**************

This is a working outline for a future section introducing NumPy to absolute beginners. If you have comments or suggestions, please don’t hesitate to reach out!



- How to install NumPy
  
  - various operating systems 

    - If you don't have Python yet, consider using Anaconda as the easiest way to get started.

      - The good thing about getting this Python distribution is is the fact that you don’t need to worry too much about separately installing NumPy or any of the major packages that you’ll be using for your data analyses, such as pandas, Scikit-Learn, etc.
    
    - If you do have Python, you can install NumPy with `conda install numpy` or `pip install numpy`
    
    - For more details, see the `Installation` section

- How to import NumPy

::

  import numpy as np 

(We shorten "numpy" to "np" in order to save time and also so that code is standardized so that anyone working with your code can easily understand and run it.)

- What is an array?

  - An array is a central data structure of the NumPy library. It's a grid of values and it contains information about the raw data, how to located an element, and how to interpret an element. All of the values in an array should be the same type and an array is indexed by a tuple of nonnegative integers. The *rank* of the array is the number of dimensions. The *shape* of the array is a tuple of integers giving the size of the array along each dimension.

  - We can initialize NumPy arrays from nested Python lists. 

  - We can access the elements in the array using square brackets. When you're accessing elements, remember that indexing starts at 0. That means that, if you want to access the first element in your array, you'll be accessing element "0".

  - How to make a NumPy array

    - To make a NumPy array, you can use the function

    ::

    np.array()

    - All you need to do to create a simple array is pass a list to it. If you choose to, you can also specify the type of data in your list. You can find more information about data types `here <https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#arrays-dtypes>`_

  ::

    import numpy as np

    # create a 1-D array
    a = np.array([1,2,3])

    # print the first element of the array
    print(a[0])
    # this will print *1*
  
  - What else might it be called?
  
  - What is its shape?

  - Can you reshape it?

  - What’s the difference between a Python List and a NumPy array? Why use NumPy?

  - What are the attributes of an array?

  - Broadcasting example

- How to create an array (ndarray object)
- How to create a basic array
- How to create an array from existing data

  - reading in a CSV

::

  import pandas as pd

  # if all columns are the same type
  x = pd.read_csv('filename.csv').values

  # otherwise, select the columns you need
  x = pd.read_csv('filename.csv', columns=['float_colname_1', ...]).values

- How to create a new array from an existing array
- How to specify the datatype
  
  - Examples of commonly used NumPy dtypes

- How to inspect the size and shape of a NumPy array
- How to check whether a list is empty or not
- How to represent missing values and infinite values
- Indexing and Slicing
- Basic array operations(np.sum, np.dot)

  - Operations on a single array

  - Unary operators

  - Binary operators

- How to compute mean, median, minimum, maximum, std, var
  
  - (include row-wise and column-wise compute)

- Sorting an array

- How to concatenate two arrays
  
  - column-wise

  - row-wise

    - np.concatenate, np.stack, np.vstack, np.hstack

- How to sort an array 
  
  - based on one (or more) columns
    
    - np.sort
    
    - np.argsort

    - np.argmin

    - np.argsort

  - based on two or more columns
    
    - np.lexsort

- How to pass a list of lists to create a 2-D array
- How to extract specific items from an array
- How to create sequences, repetitions, and random numbers

  - np.linspace
  
  - np.logspace
  
  - np.zeros

  - np.ones
  
  - np.tile

- Random Number Generation (update below to numpy.random.Generator)

  - np.random.randn
  
  - np.random.randint
  
  - np.random.random
  
  - np.random.choice
  
  - np.random.RandomState, np.random.seed

- How to get the unique items and the counts
- How to get index locations that satisfy a given condition 
- How to reverse
 
  - How to reverse the rows
 
  - How to reverse the whole array

- Reshaping and Flattening multidimensional arrays
  
  - flatten vs ravel

- How to import and export data as a CSV
- How to save and load NumPy objects
- How to apply a function column-wise or row-wise
- How to convert a 1D array into a 2D array (how to add a new axis)

- More useful functions:

  - np.clip
  
  - np.digitize
  
  - np.bincount
  
  - np.histogram
