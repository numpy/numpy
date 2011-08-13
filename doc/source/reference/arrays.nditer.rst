.. currentmodule:: numpy

.. _arrays.nditer:

*********************
Iterating Over Arrays
*********************

The iterator object :class:`nditer`, introduced in NumPy 1.6, provides many
flexible ways to visit all the elements of one or more arrays in a systematic
fashion. This page introduces some basic ways to use the object to do
computations on arrays in Python. Since the Python exposure of :class:`nditer`
is a relatively straightforward mapping of the C API for the iterator,
these ideas may also provide help working with array iteration in C.

Single Array Iteration
======================

The most basic task that can be done with the :class:`nditer` is to
visit every element of an array. Each element is provided one by one
using the standard Python iterator interface.

.. admonition:: Example

    >>> a = np.arange(6).reshape(2,3)
    >>> for x in np.nditer(a):
    ...     print x,
    ...
    0 1 2 3 4 5

An important thing to be aware of for this iteration is that the order
is chosen to match the memory layout of the array instead of using a
standard C or Fortran ordering. This is done for access efficiency,
reflecting the idea that by default one simply wants to visit each element
without concern for a particular ordering. We can see this by iterating
over the transpose our previous array, compared to taking a copy if it
in C order.

.. admonition:: Example

    >>> a = np.arange(6).reshape(2,3)
    >>> for x in np.nditer(a.T):
    ...     print x,
    ...
    0 1 2 3 4 5

    >>> for x in np.nditer(a.T.copy(order='C')):
    ...     print x,
    ...
    0 3 1 4 2 5

The elements of both `a` and `a.T` get traversed in the same order,
namely the order they are stored in memory.

Controlling Iteration Order
===========================

There are times when it is important to visit the elements of an array
in a specific order, irrespective of the layout of the elements in memory.
The :class:`nditer` object provides an `order` parameter to control this
aspect of iteration. The default, having the behavior described above,
is order='K' to keep the existing order. This can be overridden with
order='C' for C order and order='F' for Fortran order.

.. admonition:: Example

    >>> a = np.arange(6).reshape(2,3)
    >>> for x in np.nditer(a, order='F'):
    ...     print x,
    ...
    0 3 1 4 2 5
    >>> for x in np.nditer(a.T, order='C'):
    ...     print x,
    ...
    0 3 1 4 2 5

Modifying Array Values
======================

By default, the :class:`nditer` treats the input array as a read-only
object. To modify the array elements, you must specify you want to
use either read-write or write-only mode. This is controlled using
per-operand flags.

One thing to watch out for is that regular assignment in Python is
simply changing a reference in the local or global variable dictionary.
This means that simply assigning to `x` will not place a value into
the element of the array, but will rather switch `x` from being a reference
to an array element to being a reference to the value you assigned. To
actually modify the element of the array, `x` should be indexed with
the ellipsis.

.. admonition:: Example

    >>> a = np.arange(6).reshape(2,3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> for x in np.nditer(a, op_flags=['readwrite']):
    ...     x[...] = 2 * x
    ...
    >>> a
    array([[ 0,  2,  4],
           [ 6,  8, 10]])

Using an External Loop
======================

In all the examples so far, the elements of `a` are provided by the
iterator one at a time. While this is simple and convenient, it is
not very efficient. A better approach is to move the one-dimensional
inner loop out of the iterator and into your code. This way, NumPy's
vectorized operations can be used on larger chunks of the elements
being visited.

The :class:`nditer` will try to provide chunks that are
as large as possible to the inner loop. By forcing 'C' and 'F' order,
we get different external loop sizes. This mode is enabled by specifying
a global iterator flag.

Observe that with the default of keeping native memory order, the
iterator is able to provide a single one-dimensional chunk, whereas
when forcing Fortran order, it has to provide three chunks of two
elements each.

.. admonition:: Example

    >>> a = np.arange(6).reshape(2,3)
    >>> for x in np.nditer(a, flags=['external_loop']):
    ...     print x,
    ...
    [0 1 2 3 4 5]

    >>> for x in np.nditer(a, flags=['external_loop'], order='F'):
    ...     print x,
    ...
    [0 3] [1 4] [2 5]

Tracking an Index or Multi-Index
================================

During iteration, you may want to use the index of the current
element in a computation. For example, you may want to visit the
elements of an array in memory order, but use a C-order, Fortran-order,
or multidimensional index to look up values in a different array.

The Python iterator protocol doesn't have a natural way to query these
additional values from the iterator, so we introduce an alternate syntax
for iterating with an :class:`nditer`. This syntax explicitly works
with the iterator object itself, so its properties are readily accessible
during iteration. With this looping construct, the current value is
accessible by indexing into the iterator, and the index being tracked
is the property `index` or `multi_index` depending on what is requested.

The Python interactive interpreter unfortunately prints out the
while-loop condition during each iteration of the loop. We have modified
the output in the examples using this looping construct in order to
match the behavior of the for loop-based examples.

.. admonition:: Example

    >>> a = np.arange(6).reshape(2,3)
    >>> it = np.nditer(a, flags=['f_index'])
    >>> while not it.finished:
    ...     print "%d <%d>" % (it[0], it.index),
    ...     it.iternext()
    ...
    0 <0> 1 <2> 2 <4> 3 <1> 4 <3> 5 <5>

    >>> it = np.nditer(a, flags=['multi_index'])
    >>> while not it.finished:
    ...     print "%d <%s>" % (it[0], it.multi_index),
    ...     it.iternext()
    ...
    0 <(0, 0)> 1 <(0, 1)> 2 <(0, 2)> 3 <(1, 0)> 4 <(1, 1)> 5 <(1, 2)>

    >>> it = np.nditer(a, flags=['multi_index'], op_flags=['writeonly'])
    >>> while not it.finished:
    ...     it[0] = it.multi_index[1] - it.multi_index[0]
    ...     it.iternext()
    ...
    >>> a
    array([[ 0,  1,  2],
           [-1,  0,  1]])

Tracking an index or multi-index is incompatible with using an external
loop, because it requires a different index value per iteration. If
you try to combine these flags, the :class:`nditer` object will
raise an exception

.. admontion:: Example

    >>> a = np.zeros((2,3))
    >>> it = np.nditer(a, flags=['c_index', 'external_loop'])
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: Iterator flag EXTERNAL_LOOP cannot be used if an index or multi-index is being tracked

Buffering the Array Elements
============================

When forcing an iteration order, we observed that the external loop
option may provide the elements in smaller chunks because the elements
can't be visited in the appropriate order with a constant stride.
When writing C code, this is generally fine, however in pure Python code
this causes significant overhead.

By enabling buffering mode, the chunks provided by the iterator to
the inner loop can be made larger, significantly reducing the overhead
of the Python interpreter. In the example forcing Fortran iteration order,
the inner loop gets to see all the elements when buffering is enabled.

.. admonition:: Example

    >>> a = np.arange(6).reshape(2,3)
    >>> for x in np.nditer(a, flags=['external_loop'], order='F'):
    ...     print x,
    ...
    [0 3] [1 4] [2 5]

    >>> for x in np.nditer(a, flags=['external_loop','buffered'], order='F'):
    ...     print x,
    ...
    [0 3 1 4 2 5]

Iterating as a Specific Data Type
=================================

There are times when it is necessary to treat an array as a different
data type than it is stored as. For instance, one may want to do all
computations on 64-bit floats, even if the arrays being manipulated
are 32-bit floats.

There are two mechanisms which allow this to be done, temporary copies
and buffering mode. With temporary copies, a copy of the entire array
is made, then iteration is done in the copy. Write access is permitted
through a mode which updates the original array after all the iteration
is complete. The major drawback of temporary copies is that the temporary
copy may consume a large amount of memory, particular if the iteration
data type has a larger itemsize than the original one.

Buffering mode mitigates the memory usage issue and is more cache-friendly
than making temporary copies. Except for special cases, where the whole
array is needed at once outside the iterator, buffering is recommended
over temporary copying. Within NumPy, buffering is used by the ufuncs and
other functions to support flexible inputs with minimal memory overhead.

In our examples, we will treat the input array with a complex data type,
so that we can take square roots of negative numbers. Without enabling
copies or buffering mode, the iterator will raise an exception if the
data type doesn't match precisely.

.. admonition:: Example

    >>> a = np.arange(6).reshape(2,3) - 3
    >>> for x in np.nditer(a, op_dtypes=['complex128']):
    ...     print np.sqrt(x),
    ...
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: Iterator operand required copying or buffering, but neither copying nor buffering was enabled

In copying mode, 'copy' is specified as a per-operand flag. This is
done to provide control in a per-operand fashion. Buffering mode is
specified as a global iterator flag.

.. admonition:: Example

    >>> a = np.arange(6).reshape(2,3) - 3
    >>> for x in np.nditer(a, op_flags=['readonly','copy'],
    ...                 op_dtypes=['complex128']):
    ...     print np.sqrt(x),
    ...
    1.73205080757j 1.41421356237j 1j 0j (1+0j) (1.41421356237+0j)

    >>> for x in np.nditer(a, flags=['buffered'], op_dtypes=['complex128']):
    ...     print np.sqrt(x),
    ...
    1.73205080757j 1.41421356237j 1j 0j (1+0j) (1.41421356237+0j)

The iterator uses NumPy's casting rules to determine whether a specific
conversion is permitted. By default, it enforces 'safe' casting. This means,
for example, that it will raise an exception if you try to treat a
64-bit float array as a 32-bit float array. In many cases, the rule
'same_kind' is the most reasonable rule to use, since it will allow
conversion from 64 to 32-bit float, but not from float to int or from
complex to float.

.. admonition:: Example

    >>> a = np.arange(6.)
    >>> for x in np.nditer(a, flags=['buffered'], op_dtypes=['float32']):
    ...     print x,
    ...
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: Iterator operand 0 dtype could not be cast from dtype('float64') to dtype('float32') according to the rule 'safe'

    >>> for x in np.nditer(a, flags=['buffered'], op_dtypes=['float32'],
    ...                 casting='same_kind'):
    ...     print x,
    ...
    0.0 1.0 2.0 3.0 4.0 5.0

    >>> for x in np.nditer(a, flags=['buffered'], op_dtypes=['int32'], casting='same_kind'):
    ...     print x,
    ...
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: Iterator operand 0 dtype could not be cast from dtype('float64') to dtype('int32') according to the rule 'same_kind'

One thing to watch out for is conversions back to the original data
type when using a read-write or write-only operand. A common case is
to implement the inner loop in terms of 64-bit floats, and use 'same_kind'
casting to allow the other floating-point types to be processed as well.
While in read-only mode, an integer array could be provided, read-write
mode will raise an exception because conversion back to the array
would violate the casting rule.

.. admonition:: Example

    >>> a = np.arange(6)
    >>> for x in np.nditer(a, flags=['buffered'], op_flags=['readwrite'],
    ...                 op_dtypes=['float64'], casting='same_kind'):
    ...     x[...] = x / 2.0
    ...
    Traceback (most recent call last):
      File "<stdin>", line 2, in <module>
    TypeError: Iterator requested dtype could not be cast from dtype('float64') to dtype('int64'), the operand 0 dtype, according to the rule 'same_kind'
