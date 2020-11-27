.. _how-to-verify-bug:

##############################################################################
How to verify your first NumPy bug
##############################################################################

This how-to will go through the process of verifying a
`bug <https://github.com/numpy/numpy/issues/16354>`_ that was submitted
on the Github issues page.

Bug report:

::

    np.polymul returns an object with type np.float64 when one argument is all
    zero, and both arguments have type np.int64 or np.float32. Something
    similar happens with all zero np.complex64 giving result type np.complex128.
    
    This doesn't happen with non-zero arguments; there the result is as 
    expected.
    
    This bug isn't present in np.convolve.
    
    Reproducing code example:
    >>> import numpy as np
    >>> np.__version__
    '1.18.4'
    >>> a = np.array([1,2,3])
    >>> z = np.array([0,0,0])
    >>> np.polymul(a.astype(np.int64), a.astype(np.int64)).dtype
    dtype('int64')
    >>> np.polymul(a.astype(np.int64), z.astype(np.int64)).dtype
    dtype('float64')
    >>> np.polymul(a.astype(np.float32), z.astype(np.float32)).dtype
    dtype('float64')
    >>> np.polymul(a.astype(np.complex64), z.astype(np.complex64)).dtype
    dtype('complex128')
    Numpy/Python version information:
    >>> import sys, numpy; print(numpy.__version__, sys.version)
    1.18.4 3.7.5 (default, Nov  7 2019, 10:50:52) 
    [GCC 8.3.0]

******************************************************************************
1. Setup a virtual environment
******************************************************************************

Create a new directory, enter into it and set up a virtual environment using
your preferred method. For example, this is how to do it using `virtualenv`:

::

    virtualenv venv_np_bug
    source venv_np_bug/bin/activate

******************************************************************************
2. Install the NumPy version mentioned in the bug report
******************************************************************************

If the bug was more complex, you may need to build the NumPy version they
referenced :ref:`from source <building-from-source>`. But for this bug,
a pre-built wheel installed via `pip` will suffice:


::

    pip install numpy==1.18.4

******************************************************************************
3. Confirm their bug exists
******************************************************************************

Their claim is simple enough: the wrong dtype is returned 
if one of the inputs of the method `numpy.polymul` is a zero array. Open up a
Python terminal and type their code in and see if you can reproduce the
bug:

::

    >>> import numpy as np
    >>> np.__version__
    '1.18.4'
    >>> a = np.array([1,2,3])
    >>> z = np.array([0,0,0])
    >>> np.polymul(a.astype(np.int64), a.astype(np.int64)).dtype
    dtype('int64')
    >>> np.polymul(a.astype(np.int64), z.astype(np.int64)).dtype
    dtype('float64')
    >>> np.polymul(a.astype(np.float32), z.astype(np.float32)).dtype
    dtype('float64')
    >>> np.polymul(a.astype(np.complex64), z.astype(np.complex64)).dtype
    dtype('complex128')


******************************************************************************
4. See if the bug still exists on the latest version of NumPy
******************************************************************************

Now that the bug is confirmed, you could try and solve it by opening a PR.
Although for this specific bug, it has already been solved 
`here <https://github.com/numpy/numpy/pull/17577>`_. To confirm,
first uninstall the old version of NumPy and install the latest one::

    pip uninstall numpy
    pip install numpy


Now run the code again and confirm that the bug was solved.

::

    >>> import numpy as np
    >>> np.__version__
    '1.18.4'
    >>> a = np.array([1,2,3])
    >>> z = np.array([0,0,0])
    >>> np.polymul(a.astype(np.int64), a.astype(np.int64)).dtype
    dtype('int64')
    >>> np.polymul(a.astype(np.int64), z.astype(np.int64)).dtype
    dtype('int64')
    >>> np.polymul(a.astype(np.float32), z.astype(np.float32)).dtype
    dtype('float32')
    >>> np.polymul(a.astype(np.complex64), z.astype(np.complex64)).dtype
    dtype('complex64')

******************************************************************************
5. What to do next
******************************************************************************

Go to the `NumPy GitHub issues page <https://github.com/numpy/numpy/issues>`_
and see if you can confirm the existence of any other bugs. If you can, comment
on the issue saying so. Doing this helps the NumPy developers know if more than
one user is experiencing the issue.
