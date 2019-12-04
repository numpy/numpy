=========================
NEP 27 — Zero Rank Arrays
=========================

:Author: Alexander Belopolsky (sasha), transcribed Matt Picus <matti.picus@gmail.com>
:Status: Final
:Type: Informational
:Created: 2006-06-10
:Resolution: https://mail.python.org/pipermail/numpy-discussion/2018-October/078824.html

.. note ::

    NumPy has both zero rank arrays and scalars. This design document, adapted
    from a `2006 wiki entry`_, describes what zero rank arrays are and why they
    exist. It was transcribed 2018-10-13 into a NEP and links were updated.
    The pull request sparked `a lively discussion`_ about the continued need
    for zero rank arrays and scalars in NumPy.

    Some of the information here is dated, for instance indexing of 0-D arrays
    now is now implemented and does not error.

Zero-Rank Arrays
----------------

Zero-rank arrays are arrays with shape=().  For example:

    >>> x = array(1)
    >>> x.shape
    ()


Zero-Rank Arrays and Array Scalars
----------------------------------

Array scalars are similar to zero-rank arrays in many aspects::


    >>> int_(1).shape
    ()

They even print the same::


    >>> print int_(1)
    1
    >>> print array(1)
    1


However there are some important differences:

* Array scalars are immutable
* Array scalars have different python type for different data types

Motivation for Array Scalars
----------------------------

Numpy's design decision to provide 0-d arrays and array scalars in addition to
native python types goes against one of the fundamental python design
principles that there should be only one obvious way to do it.  In this section
we will try to explain why it is necessary to have three different ways to
represent a number.

There were several numpy-discussion threads:


* `rank-0 arrays`_ in a 2002 mailing list thread.
* Thoughts about zero dimensional arrays vs Python scalars in a `2005 mailing list thread`_]

It has been suggested several times that NumPy just use rank-0 arrays to
represent scalar quantities in all case.  Pros and cons of converting rank-0
arrays to scalars were summarized as follows:

- Pros:

  - Some cases when Python expects an integer (the most
    dramatic is when slicing and indexing a sequence:
    _PyEval_SliceIndex in ceval.c) it will not try to
    convert it to an integer first before raising an error.
    Therefore it is convenient to have 0-dim arrays that
    are integers converted for you by the array object.

  - No risk of user confusion by having two types that
    are nearly but not exactly the same and whose separate
    existence can only be explained by the history of
    Python and NumPy development.

  - No problems with code that does explicit typechecks
    ``(isinstance(x, float)`` or ``type(x) == types.FloatType)``. Although
    explicit typechecks are considered bad practice in general, there are a
    couple of valid reasons to use them.

  - No creation of a dependency on Numeric in pickle
    files (though this could also be done by a special case
    in the pickling code for arrays)

- Cons:

  - It is difficult to write generic code because scalars
    do not have the same methods and attributes as arrays.
    (such as ``.type``  or ``.shape``).  Also Python scalars have
    different numeric behavior as well.

  - This results in a special-case checking that is not
    pleasant.  Fundamentally it lets the user believe that
    somehow multidimensional homoegeneous arrays
    are something like Python lists (which except for
    Object arrays they are not).

Numpy implements a solution that is designed to have all the pros and none of the cons above.

    Create Python scalar types for all of the 21 types and also
    inherit from the three that already exist. Define equivalent
    methods and attributes for these Python scalar types.

The Need for Zero-Rank Arrays
-----------------------------

Once the idea to use zero-rank arrays to represent scalars was rejected, it was
natural to consider whether zero-rank arrays can be eliminated altogether.
However there are some important use cases where zero-rank arrays cannot be
replaced by array scalars.  See also `A case for rank-0 arrays`_ from February
2006.

* Output arguments::

    >>> y = int_(5)
    >>> add(5,5,x)
    array(10)
    >>> x
    array(10)
    >>> add(5,5,y)
    Traceback (most recent call last):
         File "<stdin>", line 1, in ?
    TypeError: return arrays must be of ArrayType

* Shared data::

    >>> x = array([1,2])
    >>> y = x[1:2]
    >>> y.shape = ()
    >>> y
    array(2)
    >>> x[1] = 20
    >>> y
    array(20)

Indexing of Zero-Rank Arrays
----------------------------

As of NumPy release 0.9.3, zero-rank arrays do not support any indexing::

    >>> x[...]
    Traceback (most recent call last):
      File "<stdin>", line 1, in ?
    IndexError: 0-d arrays can't be indexed.

On the other hand there are several cases that make sense for rank-zero arrays.

Ellipsis and empty tuple
~~~~~~~~~~~~~~~~~~~~~~~~

Alexander started a `Jan 2006 discussion`_ on scipy-dev
with the following proposal:

    ... it may be reasonable to allow ``a[...]``.  This way
    ellipsis can be interpereted as any number of  ``:`` s including zero.
    Another subscript operation that makes sense for scalars would be
    ``a[...,newaxis]`` or even ``a[{newaxis, }* ..., {newaxis,}*]``, where
    ``{newaxis,}*`` stands for any number of comma-separated newaxis tokens.
    This will allow one to use ellipsis in generic code that would work on
    any numpy type.

Francesc Altet supported the idea of ``[...]`` on zero-rank arrays and
`suggested`_ that ``[()]`` be supported as well.

Francesc's proposal was::

    In [65]: type(numpy.array(0)[...])
    Out[65]: <type 'numpy.ndarray'>

    In [66]: type(numpy.array(0)[()])   # Indexing a la numarray
    Out[66]: <type 'int32_arrtype'>

    In [67]: type(numpy.array(0).item())  # already works
    Out[67]: <type 'int'>

There is a consensus that for a zero-rank array ``x``, both ``x[...]`` and ``x[()]`` should be valid, but the question
remains on what should be the type of the result - zero rank ndarray or ``x.dtype``?

(Alexander)
    First, whatever choice is made for ``x[...]`` and ``x[()]`` they should be
    the same because ``...`` is just syntactic sugar for "as many `:` as
    necessary", which in the case of zero rank leads to ``... = (:,)*0 = ()``.
    Second, rank zero arrays and numpy scalar types are interchangeable within
    numpy, but numpy scalars can be use in some python constructs where ndarrays
    can't.  For example::

        >>> (1,)[array(0)]
        Traceback (most recent call last):
          File "<stdin>", line 1, in ?
        TypeError: tuple indices must be integers
        >>> (1,)[int32(0)]
        1

Since most if not all numpy function automatically convert zero-rank arrays to scalars on return, there is no reason for
``[...]`` and ``[()]`` operations to be different.

See SVN changeset 1864 (which became git commit `9024ff0`_) for
implementation of ``x[...]`` and ``x[()]`` returning numpy scalars.

See SVN changeset 1866 (which became git commit `743d922`_) for
implementation of ``x[...] = v`` and ``x[()] = v``

Increasing rank with newaxis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everyone who commented liked this feature, so as of SVN changeset 1871 (which became git commit `b32744e`_) any number of ellipses and
newaxis tokens can be placed as a subscript argument for a zero-rank array. For
example::

    >>> x = array(1)
    >>> x[newaxis,...,newaxis,...]
    array([[1]])

It is not clear why more than one ellipsis should be allowed, but this is the
behavior of higher rank arrays that we are trying to preserve.

Refactoring
~~~~~~~~~~~

Currently all indexing on zero-rank arrays is implemented in a special ``if (nd
== 0)`` branch of code that used to always raise an index error. This ensures
that the changes do not affect any existing usage (except, the usage that
relies on exceptions).  On the other hand part of motivation for these changes
was to make behavior of ndarrays more uniform and this should allow to
eliminate  ``if (nd == 0)`` checks altogether.

Copyright
---------

The original document appeared on the scipy.org wiki, with no Copyright notice, and its `history`_ attributes it to sasha.

.. _`2006 wiki entry`: https://web.archive.org/web/20100503065506/http://projects.scipy.org:80/numpy/wiki/ZeroRankArray
.. _`history`: https://web.archive.org/web/20100503065506/http://projects.scipy.org:80/numpy/wiki/ZeroRankArray?action=history
.. _`2005 mailing list thread`: https://sourceforge.net/p/numpy/mailman/message/11299166
.. _`suggested`: https://mail.python.org/pipermail/numpy-discussion/2006-January/005572.html
.. _`Jan 2006 discussion`: https://mail.python.org/pipermail/numpy-discussion/2006-January/005579.html
.. _`A case for rank-0 arrays`: https://mail.python.org/pipermail/numpy-discussion/2006-February/006384.html
.. _`rank-0 arrays`: https://mail.python.org/pipermail/numpy-discussion/2002-September/001600.html
.. _`9024ff0`: https://github.com/numpy/numpy/commit/9024ff0dc052888b5922dde0f3e615607a9e99d7
.. _`743d922`: https://github.com/numpy/numpy/commit/743d922bf5893acf00ac92e823fe12f460726f90
.. _`b32744e`: https://github.com/numpy/numpy/commit/b32744e3fc5b40bdfbd626dcc1f72907d77c01c4
.. _`a lively discussion`: https://github.com/numpy/numpy/pull/12166
