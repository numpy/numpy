<<<<<<< HEAD
****************************************
NumPy: The Absolute Basics for Beginners
****************************************
=======
**************
NumPy: The Absolute Basics for Beginners
**************
>>>>>>> absolute-beginners

This is a working outline for a future section introducing NumPy to absolute beginners. If you have comments or suggestions, please don’t hesitate to reach out!



- How to install NumPy
  
<<<<<<< HEAD
  - various operating systems
    - If you don't have Python yet, consider using Anaconda as the easiest way to get started
    - If you do have Python, you can install NumPy with `conda install numpy` or `pip install numpy`
    - For more details, see the `Installation` section
=======
  - various operating systems 
>>>>>>> absolute-beginners

- How to import NumPy

- What is an array
  
  - What else might it be called
  
  - What is its shape

  - Can you reshape it?

  - What’s the difference between a Python List and a NumPy array? Why use NumPy?

  - What are the attributes of an array?

  - Broadcasting example

- How to create an array (ndarray object)
- How to create a basic array
- How to create an array from existing data
<<<<<<< HEAD

  - reading in a CSV

::

  import pandas as pd

  # if all columns are the same type
  x = pd.read_csv('filename.csv').values

  # otherwise, select the columns you need
  x = pd.read_csv('filename.csv', columns=['float_colname_1', ...]).values


=======
>>>>>>> absolute-beginners
- How to create a new array from an existing array
- How to specify the datatype
  
  - Examples of commonly used NumPy dtypes

- How to inspect the size and shape of a NumPy array
- How to check whether a list is empty or not
- How to represent missing values and infinite values
- Indexing and Slicing
<<<<<<< HEAD
- Basic array operations (np.sum, np.dot)
=======
- Basic array operations (np.sum, np.dot, np.append, np.diff
>>>>>>> absolute-beginners

  - Operations on a single array

  - Unary operators

  - Binary operators

<<<<<<< HEAD
- How to compute mean, median, minimum, maximum, std, var)
  
  - (include row-wise and column-wise compute)

- Sorting an array
=======
  - Universal functions

- How to compute mean, minimum, maximum, cumulative sum
  
  - (include row-wise and column-wise compute)

-S orting an array
>>>>>>> absolute-beginners

- How to concatenate two arrays
  
  - column-wise

<<<<<<< HEAD
  - row-wise

    - np.concatenate, np.stack, np.vstack, np.hstack
=======
- row-wise

    - np.concatenate, np.vstack, np.hstack, np.r_, np.c_
>>>>>>> absolute-beginners

- How to sort an array 
  
  - based on one (or more) columns
    
    - np.sort
    
    - np.argsort

<<<<<<< HEAD
    - np.argmin

    - np.argsort

=======
>>>>>>> absolute-beginners
  - based on two or more columns
    
    - np.lexsort

<<<<<<< HEAD
- How to pass a list of lists to create a 2-D array
=======
- How to pass a list of lists to create a matrix
>>>>>>> absolute-beginners
- How to extract specific items from an array
- How to create sequences, repetitions, and random numbers

  - np.linspace
  
  - np.logspace
  
  - np.zeros

  - np.ones
  
  - np.tile

<<<<<<< HEAD
- Random Number Generation (update below to numpy.random.Generator)

=======
>>>>>>> absolute-beginners
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
<<<<<<< HEAD
=======
- Working with dates and times
  
  - creating a date time object
  
  - removing time from date
  
  - create individual units of time
  
  - convert back to a string
  
  - filter business days
  
  - creating a sequence of dates
>>>>>>> absolute-beginners

- More useful functions:

  - np.clip
  
  - np.digitize
  
  - np.bincount
  
  - np.histogram
