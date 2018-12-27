======================================================
NEP 28 — High Level Data Types and Universal Functions
======================================================

:Author: Stephan Hoyer
:Status: Draft
:Type: Standards Track
:Created: 2018-12-15


Abstract
========

This document outlines what NumPy needs to make it possible to write
fully-featured data types from Python. We focus on universal functions, since
these are NumPy's main abstration for data type dependent functionality.

This arose out of discussion at the NumPy developer meeting in Berkeley,
California on November 30, 2018. Thanks to Eric Wieser, Matti Pincus, Charles
Harris, and Travis Oliphant for sharing their thoughts on parts of this
proposal.

The dtype abstraction
---------------------

Data types ("dtype") in NumPy idenfies the type of the data stored in arrays
(e.g., float vs integer). The power of dtypes is that they allows for a clean
separation (at least in principle) between shape and data dependent
functionality:

1. "Shape functions" depend on the shapes of arrays, but are mostly agnostic to
   the underlying data. Examples: ``concatenate``, ``reshape``, indexing.
2. "Dtype functions" depend on the data type but are mostly agnostic to shapes.
   Many but not all such functions in NumPy are `universal functions`_ ("ufuncs").
   Examples: ``add``, ``sin``, ``sum``.

NumPy up to 1.16 has poor support for custom dtypes. Defining dtype-dependent
functionality requires  writing inner loops in C, and even the C API is
insufficiently flexible for many use cases (e.g., there is no support for
functions that depend on metadata stored on dtypes).

.. _universal functions: https://docs.scipy.org/doc/numpy/reference/ufuncs.html

Types of dtypes
---------------

NumPy comes with many builtin dtypes, and there are also many `many
usecases`_  for user-defined dtypes. These fall into several categories:

1. Low level "physical" dtypes written in C make use of NumPy's low-level
   machinery. Today this is the only sort of dtype that is possible to write.
   Examples: current NumPy dtypes, `quaternion`_ , novel floating point types.

2. Low-level dtypes that need some mechanism for allocation/deallocation of
   memory associated with NumPy arrays, because they don't have fixed size
   elements. These dtypes need arrays of pointers, along with separate memory
   storage, i.e., similar to the existing object dtype, but manage memory
   themselves instead of deferring entirely to Python. Example use cases: variable
   width strings, polygons/shapes, ragged arrays, big integers.

3. High-level "logical" dtypes that could be defined by dtype written in Python
   that reuse fast operations written for low-level dtypes. Example use cases:
   physical units, datetimes, missing values (either using a sentinel value or a
   separate mask), categoricals/enumerations, encoded fixed-width text.

These are all valuable, but here we want to focus on category (3). We think
this sort of dtype has the largest need, and would have the largest beneficial
impact on the broader NumPy ecosystem.

This NEP intentionally does not address questions of how NumPy's low-level
dtype machinery will need to be updated to make it possible to write high-level
logical dtypes. These will be the subject another NEP.

.. _`many usecases`: https://github.com/numpy/numpy/wiki/Dtype-Brainstorming)
.. _`quaternion`: https://github.com/moble/quaternion

Why aren't duck arrays or subclasses enough?
--------------------------------------------

"Duck arrays" are great and we should support them better in NumPy. See
`NEP-18`_ for an overview and discussion. Most functionality envisioned for
dtypes could also be implemented by duck arrays, but duck arrays require a lot
more work: you need to redefine all of the shape functions as well as the dtype
functions.

Subclasses of NumPy arrays are widely used and, at least in principle, could
satisfy the use-case of extending dtype-specific functions without rewriting
shape-dependent functions. Unfortunately, subclasses are a little *too*
flexible: it's easy to write a subclass that violates the `Liskov substitution
principle`_. NumPy itself includes several examples of `ndarray` subclasses
that violate substitutabiltiy, either by changing shape semenatics (for
`np.matrix`) or the behavior of methods (for `np.ma.MaskedArray`).

Dtypes provide just the right amount of flexibility to define a new data type,
while allowing them to be reliably used from third-party libraries. You need to
define new functionality related to the dtype, but shape functions are
guaranteed to work exactly like standard NumPy arrays.

.. _`NEP-18`: nep-0022-ndarray-duck-typing-overview
.. _`Liskov substitution principle`: https://en.wikipedia.org/wiki/Liskov_substitution_principle

Defining a dtype from Python
============================

Dtype customized functions should be ufuncs
-------------------------------------------

The core idea of universal functions is that they are  data-dependent but shape
agnostic. They use shared function signatures allows them to be processed in
standard ways, e.g., with `__array_ufunc__`_.

We should double-down on `ufuncs`, by making all data-dependent behavior in
NumPy use `ufuncs`. See `issue 12514`_ for a list of things that need to
change. This isn't to say that every data-dependent function in NumPy needs to
be ufunc: in many cases, functions have esoteric enough signatures (e.g., for
shape handling) that they can't fit into the ufunc model. But we should write
ufuncs for all the data-dependent functionality within NumPy, and implementing
ufuncs for a new dtype should be NumPy's extension mechanism for dtype-specific
unctionality.

.. _`issue 12514`: https://github.com/numpy/numpy/issues/12514

A proposed interface
--------------------

Note: This NEP presumes that dtypes will be rewritten as a metaclasses, so
NumPy scalars will become instances of dtypes. This will be the subject of
another NEP.

Writing a high-level Python dtype should simply entail writing a new Python
type that implements a standard interface. This interface should specify:

- **Metadata:** The minimum that NumPy needs to know about a custom dtype so
  memory allocation and shape operations work.

  - ``itemsize``: how many bytes does each dtype element consume?
  - ``alignment`` (should be ``itemsize`` or smaller).
- **Casting rules:** How do we convert to and from this dtype?

  - What is the common dtype that a set of dtypes including this dtype can be
    cast to?
  - Cast another dtype to this dtype (with either safe or unsafe casting).
  - Should array indexing give a dtype scalar or another type? (as supported
    by ``dtype=object``, which uses ``NPY_USE_GETITEM``)
  - What is the "builtin" type returned by ``ndarray.item()``?
- **Dtype-specific functions**: By design, these should be restricted to ufuncs.

  - Dtypes need a mechanism like ``__array_ufunc__`` to override existing
    ufuncs (see `__dtype_ufunc__` below)
  - It should also be easy to write new user-defined ufuncs, e.g.,
    specifically for this dtype. User defined ufuncs should be preferred to
    writing other types of functions.
- **Printing**: The dtype should specify how scalar elements are displayed,
  e.g., via the ``__repr__`` method.

By design, it is impossible to override arbitrary NumPy functions that act on
shapes, e.g., ``np.concatenate``. These should not vary in a dtype dependent
fashion.

Dtype specific ufunc overrides
==============================

We propose a more restricted variant of ``__array_ufunc__`` (only for high
level Python dtypes) that restricts itself to **not** handle duckarrays, which
we'll tentatively call ``__dtype__ufunc__``.

Unlike ``__array_ufunc__``, calling ufunc overrides ``__dtype_ufunc__`` should
happen at a lower level in the ufunc machinery:
- Inputs are guaranteed to be NumPy arrays.
- Outputs are required to be NumPy arrays with the expected shapes for the operation.

However, ``__dtype_func__`` overrides happens at a higher level than NumPy's
existing ufunc implementations:

- You can wrap an existing inner loop.
- You don't need to write the C type resolver.
- You don't need to pre-specify how the implementation is selected -- this can
  be deferred to the types.

Drawbacks:

- Multi-level dispatch complexity (``__array_ufunc__``, ``__dtype_ufunc__`` and
  NumPy's internal thing).
- Not as well factorized as casting + low-level loops.

Example usage
=============

Consider datetime and timedelta dtype like NumPy's datetime64/timedelta64.

Most operations could be implemented simply by casting to int64 and calling
another ufunc on the int64 data, e.g., for ``np.sub``::

    class MyDatetime(metaclass=np.dtype):
        @classmethod
        def __dtype_ufunc__(cls, ufunc, method, *inputs, **kwargs):
            if method != '__call__':
                return NotImplemented
            if ufunc is np.sub:
                a, b = inputs
                if isinstance(a, cls) and isinstance(b, cls):
                    return (a.view(np.int64) - b.view(np.int64)).view(MyTimedelta)
                elif isinstance(b, MyTimdelta):
                    return (a.view(np.int64) - b.view(np.int64)).view(MyDatetime)
                else:
                    return NotImplemented
            # implement other ufuncs
            return NotImplemented

    class MyTimedelta(metaclass=np.dtype):
        ...


How NumPy calls ``__dtype_ufunc__``
===================================

NumPy should check for ``__dtype_ufunc__`` attributes after looking for
``__array_ufunc__`` overrides, but before builtin ufunc implementations,
*e.g.*::

    def implement_dtype_ufunc(ufunc, method, *inputs, **kwargs):
        outputs = kwargs.get('out', ())
        arrays = inputs + outputs

        # dtype dispatching
        dtypes = [item.dtype
                  for item in arrays
                  if hasattr(item.dtype, '__dtype_ufunc__')]
        if dtypes:
            for dtype in dtypes:
                # note: each element in inputs is a numpy array
                # or subclass
                result = dtype.__dtype_ufunc__(
                    ufunc, method, *inputs, **kwargs)
                if result is not NotImplemented:
                    check_result(result, ufunc, inputs, kwargs)
                    return result
            raise TypeError('dtypes did not implement ufunc')

        # base ndarray implementation
        return getattr(ufunc, method)(*items, **kwargs)

As part of calling ``__dtype_ufunc__`` overrides, NumPy should verify that the
custom ufunc implementation honors appropriate invariants::

    def check_result(result, ufunc, inputs):
        # various consistency checks for the result

        if type(result) is not tuple:
            result = (result,)
            
        if len(result) != ufunc.nout:
            raise ValueError('wrong number of outputs')

        for x in result:
            if not isinstance(x, ndarray):
                raise TypeError('wrong result type')
        
        # TODO: handle gufunc shapes
        expected_shape = broadcast_arrays(*inputs).shape
        for expected_shape, res in zip(shapes, result):
            if expected_shape != res.shape:
                raise ValueError('wrong shape')

Defining new universal functions from Python
--------------------------------------------

Most dtypes need new functions, beyond those that already exist as ufuncs in
NumPy. For example, our new datetime type should have functions for doing
custom datetime conversions.

Logically, almost all of these operations are element-wise, so they are a good
fit for NumPy's ufunc model. But right now it's hard to write ufuncs: you need
to define the inner loops at a C level, and sometimes even write or modify
NumPy's internal "type resolver" function that determines the proper output
type and inner loop function to use given certain input types (e.g., NumPy has
hard-coded support for ``datetime64`` in the type resolver for ``np.add``). For
user-defined dtypes written in Python to be usable, it should be possible write
user-defined ufuncs in Python, too.

Use cases
=========

There are least three use-cases for writing ufuncs in Python:

1. Creating real ufuncs from element-wise functions, e.g., like
``np.vectorize`` but actually creating a ufunc. This will not be terribly
useful because it is painfully slow to do inner loops in Python.
2. Creating real ufuncs from vectorized functions written in Python that don't
do broadcasting but are defined on vectors, i.e., writing the "inner loop" for
a ufunc from Python instead of C.
3. Marking already vectorized functions as ufuncs, so they can be overriden and
manipulated in a generic way from ``__dtype_ufunc__`` or ``__array_ufunc__``.
This provides useful introspection options for third-party libraries to build
upon, e.g., ``dask.array`` can automatically determine how to parallelize such a
ufunc.

For usable user-defined ufuncs, case (2) is probably most important. There are
lots of examples of performant vectorized functions in user code, but with the
exception of trivial cases where non-generalized NumPy ufuncs are wrapped, most
of these don't handle the full generality of NumPy's ufuncs.

For NumPy itself, case (3) could be valuable: we have lots of non-ufuncs that
could logically fit into the ufunc model, e.g., ``argmin``, ``median``,
``sort``, ``where``, etc.

Note: ``numba.vectorize`` is does not produce a ufunc currently, but it should be.

Proposed interfaces
===================

A ufunc decorator should check args, and do broadcasting/reshaping such that
the ufunc implementation only needs to handle arrays with one more dimensions
than the number of "core dimensions" in the `gufunc signature`_.
For example::

    @ufunc(signature='()->()')
    def dayofyear(dates):
        # dates is a 1D numpy array
        return dates.view(np.int64) % 365

or perhaps supporting multiple loops::

    # the gufunc signature shows nin/nout and dimensionality
    @ufunc(signature='()->()')
    def dayofyear(dates):
        """Used just for documentation."""

    @dayofyear.define_loop([MyDatetime, MyDatetime])
    def dayofyear(dates):
        # dates is a 1D numpy array
        return dates.view(np.int64) % 365

    @dayofyear.define_loop([np.generic, MyDatetime])
    def dayofyear(dates):
        ...
    
    # or extracting the dtypes from annotations:
    @dayofyear.define_loop
    def dayofyear(dates: Array[np.generic]) -> Array[MyDatetime]:
        ...

This is doing three things:

- Syntactic sugar for creating a ufunc
- Syntactic sugar for registering ufunc inner loops
- Conversion of Python inner loops into C inner loops

Why is ``@ufunc`` different from ``vectorize``?

- True ufuncs can be overriden with ``__array_ufunc__``_ or ``__dtype_ufunc__``.
- NumPy can implement some arguments automatically (e.g., ``where``, and
  ``axis`` for gufuncs).

Changes within NumPy
====================

**TODO**: finish cleaning this up

NumPy's low-level ufunc machinery
---------------------------------

For each ufunc, we currently have:

- Type resolver function
- Casting to the resolved types
- Loops for specific dtypes

This results in hard-wired cases for new dtypes (e.g., ``np.datetime64``)
inside type resolver functions, which is not very extensible.

Instead, we might want:

- Type resolver protocol (like ``__dtype_ufunc__`` but without the overhead of
  Python calls) finds a dtype that implements the ufunc for all the given
  argument dtypes
- Do the dtype specific casting and inner loops

We will want to default to using NumPy's current type resolving protocol for
current builtin dtypes/ufuncs, i.e., by writing a generic version of
``__low_level_dtype_ufunc__`` to set on builtin dtypes.

Rewriting existing NumPy functions
----------------------------------

- There are a handful of dtype specific functions that aren't ufuncs and
  couldn't currently fit into ufuncs:

  - Some of these functions use custom keyword arguments, which currently can't
    be used on ufuncs (e.g., ``ddof`` on ``np.std``):

    - You want be able provide positional arguments as keyword arguments.
    - You might want to vectorize across keyword arguments (or not)

  - Others have a shape signature that doesn't fit into gufuncs:

    - Gufuncs could potentially be extended to handle aggregations like
      ``np.mean``, or perhaps we could define these as ufuncs that have a
      ``reduce`` method but no ``__call__`` method.
    - Various linear algebra functions (e.g., ``np.linalg.solve``) have their
      own strange casting rules. If we want to support these, we will need some
      dtype equivalent version of ``__array_function__``.

- There are some existing functions inside NumPy that could make use of these
  mechanisms:

  - NumPy's datetime functions (e.g., ``np.busday_count``)
  - Needs to be a gufuncs: ``sort``, ``mean``, ``median`` etc.
    - `mean` will need new axis rules.
  - Functions like ``np.where`` are vectorized like ufuncs, but it can use a
    generic (non dtype-dependent) inner loop.

    - Challenge: this is only true in one branch. Solution: make a new ufunc
      that gets exposed publically (even if just ``np.where.ufunc``).
  - Likewise, linear algebra functions use multiple gufuncs internally. Could
    potentially expose these publically. Or: could rewrite them as a single
    gufunc with custom loop selection.

Appendix
========

References
----------

- pandas `ExtensionArray interface <https://github.com/pandas-dev/pandas/blob/5b0610b875476a6f3727d7e9bedb90d370c669b5/pandas/core/arrays/base.py>`
- Dtype `brainstorming session <https://github.com/numpy/numpy/wiki/Dtype-Brainstorming>`
  from SciPy

The current interface of dtypes in NumPy
----------------------------------------

.. code-block:: python

    class DescrFlags(IntFlags):
        # The item must be reference counted when it is inserted or extracted.
        ITEM_REFCOUNT   = 0x01
        # Same as needing REFCOUNT
        ITEM_HASOBJECT  = 0x01
        # Convert to list for pickling
        LIST_PICKLE     = 0x02
        # The item is a POINTER 
        ITEM_IS_POINTER = 0x04
        # memory needs to be initialized for this data-type
        NEEDS_INIT      = 0x08
        # operations need Python C-API so don't give-up thread.
        NEEDS_PYAPI     = 0x10
        # Use f.getitem when extracting elements of this data-type
        USE_GETITEM     = 0x20
        # Use f.setitem when setting creating 0-d array from this data-type
        USE_SETITEM     = 0x40
        # A sticky flag specifically for structured arrays
        ALIGNED_STRUCT  = 0x80

    class current_dtype(object):
        itemsize: int
        alignment: int
        
        byteorder: str
        flags: DescrFlags
        metadata: ...  # unknown
        
        # getters
        hasobject: bool
        isalignedstruct: bool
        isbuiltin: bool
        isnative: bool
        
        
        def newbyteorder(self) -> current_dtype: ...
        
        # to move to a structured dtype subclass
        names: Tuple[str]
        fields: Dict[str, Union[
        Tuple[current_dtype, int],
        Tuple[current_dtype, int, Any]
        ]]
        
        # to move to a subarray dtype subclass
        subdtype: Optional[Tuple[dtype, Tuple[int,...]]]
        shape: Tuple[int]
        base: current_dtype
        
        # to deprecate
        type: Type  # merge with cls
        kind: str
        num: int
        str: str
        name: str
        char: str
        descr: List[...]

.. _`__array_ufunc__`: nep-0013-ufunc-overrides
.. _gufunc signature: https://docs.scipy.org/doc/numpy-1.15.0/reference/c-api.generalized-ufuncs.html
