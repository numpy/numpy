=============================================================
NEP 16 — An abstract base class for identifying "duck arrays"
=============================================================

:Author: Nathaniel J. Smith <njs@pobox.com>
:Status: Withdrawn
:Type: Standards Track
:Created: 2018-03-06
:Resolution: https://github.com/numpy/numpy/pull/12174

.. note::

    This NEP has been withdrawn in favor of the protocol based approach
    described in
    `NEP 22 <nep-0022-ndarray-duck-typing-overview.html>`__

Abstract
--------

We propose to add an abstract base class ``AbstractArray`` so that
third-party classes can declare their ability to "quack like" an
``ndarray``, and an ``asabstractarray`` function that performs
similarly to ``asarray`` except that it passes through
``AbstractArray`` instances unchanged.


Detailed description
--------------------

Many functions, in NumPy and in third-party packages, start with some
code like::

   def myfunc(a, b):
       a = np.asarray(a)
       b = np.asarray(b)
       ...

This ensures that ``a`` and ``b`` are ``np.ndarray`` objects, so
``myfunc`` can carry on assuming that they'll act like ndarrays both
semantically (at the Python level), and also in terms of how they're
stored in memory (at the C level). But many of these functions only
work with arrays at the Python level, which means that they don't
actually need ``ndarray`` objects *per se*: they could work just as
well with any Python object that "quacks like" an ndarray, such as
sparse arrays, dask's lazy arrays, or xarray's labeled arrays.

However, currently, there's no way for these libraries to express that
their objects can quack like an ndarray, and there's no way for
functions like ``myfunc`` to express that they'd be happy with
anything that quacks like an ndarray. The purpose of this NEP is to
provide those two features.

Sometimes people suggest using ``np.asanyarray`` for this purpose, but
unfortunately its semantics are exactly backwards: it guarantees that
the object it returns uses the same memory layout as an ``ndarray``,
but tells you nothing at all about its semantics, which makes it
essentially impossible to use safely in practice. Indeed, the two
``ndarray`` subclasses distributed with NumPy – ``np.matrix`` and
``np.ma.masked_array`` – do have incompatible semantics, and if they
were passed to a function like ``myfunc`` that doesn't check for them
as a special-case, then it may silently return incorrect results.


Declaring that an object can quack like an array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two basic approaches we could use for checking whether an
object quacks like an array. We could check for a special attribute on
the class::

  def quacks_like_array(obj):
      return bool(getattr(type(obj), "__quacks_like_array__", False))

Or, we could define an `abstract base class (ABC)
<https://docs.python.org/3/library/collections.abc.html>`__::

  def quacks_like_array(obj):
      return isinstance(obj, AbstractArray)

If you look at how ABCs work, this is essentially equivalent to
keeping a global set of types that have been declared to implement the
``AbstractArray`` interface, and then checking it for membership.

Between these, the ABC approach seems to have a number of advantages:

* It's Python's standard, "one obvious way" of doing this.

* ABCs can be introspected (e.g. ``help(np.AbstractArray)`` does
  something useful).

* ABCs can provide useful mixin methods.

* ABCs integrate with other features like mypy type-checking,
  ``functools.singledispatch``, etc.

One obvious thing to check is whether this choice affects speed. Using
the attached benchmark script on a CPython 3.7 prerelease (revision
c4d77a661138d, self-compiled, no PGO), on a Thinkpad T450s running
Linux, we find::

    np.asarray(ndarray_obj)      330 ns
    np.asarray([])              1400 ns

    Attribute check, success      80 ns
    Attribute check, failure      80 ns

    ABC, success via subclass    340 ns
    ABC, success via register()  700 ns
    ABC, failure                 370 ns

Notes:

* The first two lines are included to put the other lines in context.

* This used 3.7 because both ``getattr`` and ABCs are receiving
  substantial optimizations in this release, and it's more
  representative of the long-term future of Python. (Failed
  ``getattr`` doesn't necessarily construct an exception object
  anymore, and ABCs were reimplemented in C.)

* The "success" lines refer to cases where ``quacks_like_array`` would
  return True. The "failure" lines are cases where it would return
  False.

* The first measurement for ABCs is subclasses defined like::

      class MyArray(AbstractArray):
          ...

  The second is for subclasses defined like::

      class MyArray:
          ...

      AbstractArray.register(MyArray)

  I don't know why there's such a large difference between these.

In practice, either way we'd only do the full test after first
checking for well-known types like ``ndarray``, ``list``, etc. `This
is how NumPy currently checks for other double-underscore attributes
<https://github.com/numpy/numpy/blob/master/numpy/core/src/private/get_attr_string.h>`__
and the same idea applies here to either approach. So these numbers
won't affect the common case, just the case where we actually have an
``AbstractArray``, or else another third-party object that will end up
going through ``__array__`` or ``__array_interface__`` or end up as an
object array.

So in summary, using an ABC will be slightly slower than using an
attribute, but this doesn't affect the most common paths, and the
magnitude of slowdown is fairly small (~250 ns on an operation that
already takes longer than that). Furthermore, we can potentially
optimize this further (e.g. by keeping a tiny LRU cache of types that
are known to be AbstractArray subclasses, on the assumption that most
code will only use one or two of these types at a time), and it's very
unclear that this even matters – if the speed of ``asarray`` no-op
pass-throughs were a bottleneck that showed up in profiles, then
probably we would have made them faster already! (It would be trivial
to fast-path this, but we don't.)

Given the semantic and usability advantages of ABCs, this seems like
an acceptable trade-off.

..
   CPython 3.6 (from Debian)::

       Attribute check, success     110 ns
       Attribute check, failure     370 ns

       ABC, success via subclass    690 ns
       ABC, success via register()  690 ns
       ABC, failure                1220 ns


Specification of ``asabstractarray``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given ``AbstractArray``, the definition of ``asabstractarray`` is simple::

  def asabstractarray(a, dtype=None):
      if isinstance(a, AbstractArray):
          if dtype is not None and dtype != a.dtype:
              return a.astype(dtype)
          return a
      return asarray(a, dtype=dtype)

Things to note:

* ``asarray`` also accepts an ``order=`` argument, but we don't
  include that here because it's about details of memory
  representation, and the whole point of this function is that you use
  it to declare that you don't care about details of memory
  representation.

* Using the ``astype`` method allows the ``a`` object to decide how to
  implement casting for its particular type.

* For strict compatibility with ``asarray``, we skip calling
  ``astype`` when the dtype is already correct. Compare::

      >>> a = np.arange(10)

      # astype() always returns a view:
      >>> a.astype(a.dtype) is a
      False

      # asarray() returns the original object if possible:
      >>> np.asarray(a, dtype=a.dtype) is a
      True


What exactly are you promising if you inherit from ``AbstractArray``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This will presumably be refined over time. The ideal of course is that
your class should be indistinguishable from a real ``ndarray``, but
nothing enforces that except the expectations of users. In practice,
declaring that your class implements the ``AbstractArray`` interface
simply means that it will start passing through ``asabstractarray``,
and so by subclassing it you're saying that if some code works for
``ndarray``\s but breaks for your class, then you're willing to accept
bug reports on that.

To start with, we should declare ``__array_ufunc__`` to be an abstract
method, and add the ``NDArrayOperatorsMixin`` methods as mixin
methods.

Declaring ``astype`` as an ``@abstractmethod`` probably makes sense as
well, since it's used by ``asabstractarray``. We might also want to go
ahead and add some basic attributes like ``ndim``, ``shape``,
``dtype``.

Adding new abstract methods will be a bit tricky, because ABCs enforce
these at subclass time; therefore, simply adding a new
`@abstractmethod` will be a backwards compatibility break. If this
becomes a problem then we can use some hacks to implement an
`@upcoming_abstractmethod` decorator that only issues a warning if the
method is missing, and treat it like a regular deprecation cycle. (In
this case, the thing we'd be deprecating is "support for abstract
arrays that are missing feature X".)


Naming
~~~~~~

The name of the ABC doesn't matter too much, because it will only be
referenced rarely and in relatively specialized situations. The name
of the function matters a lot, because most existing instances of
``asarray`` should be replaced by this, and in the future it's what
everyone should be reaching for by default unless they have a specific
reason to use ``asarray`` instead. This suggests that its name really
should be *shorter* and *more memorable* than ``asarray``... which
is difficult. I've used ``asabstractarray`` in this draft, but I'm not
really happy with it, because it's too long and people are unlikely to
start using it by habit without endless exhortations.

One option would be to actually change ``asarray``\'s semantics so
that *it* passes through ``AbstractArray`` objects unchanged. But I'm
worried that there may be a lot of code out there that calls
``asarray`` and then passes the result into some C function that
doesn't do any further type checking (because it knows that its caller
has already used ``asarray``). If we allow ``asarray`` to return
``AbstractArray`` objects, and then someone calls one of these C
wrappers and passes it an ``AbstractArray`` object like a sparse
array, then they'll get a segfault. Right now, in the same situation,
``asarray`` will instead invoke the object's ``__array__`` method, or
use the buffer interface to make a view, or pass through an array with
object dtype, or raise an error, or similar. Probably none of these
outcomes are actually desirable in most cases, so maybe making it a
segfault instead would be OK? But it's dangerous given that we don't
know how common such code is. OTOH, if we were starting from scratch
then this would probably be the ideal solution.

We can't use ``asanyarray`` or ``array``, since those are already
taken.

Any other ideas? ``np.cast``, ``np.coerce``?


Implementation
--------------

1. Rename ``NDArrayOperatorsMixin`` to ``AbstractArray`` (leaving
   behind an alias for backwards compatibility) and make it an ABC.

2. Add ``asabstractarray`` (or whatever we end up calling it), and
   probably a C API equivalent.

3. Begin migrating NumPy internal functions to using
   ``asabstractarray`` where appropriate.


Backward compatibility
----------------------

This is purely a new feature, so there are no compatibility issues.
(Unless we decide to change the semantics of ``asarray`` itself.)


Rejected alternatives
---------------------

One suggestion that has come up is to define multiple abstract classes
for different subsets of the array interface. Nothing in this proposal
stops either NumPy or third-parties from doing this in the future, but
it's very difficult to guess ahead of time which subsets would be
useful. Also, "the full ndarray interface" is something that existing
libraries are written to expect (because they work with actual
ndarrays) and test (because they test with actual ndarrays), so it's
by far the easiest place to start.


Links to discussion
-------------------

* https://mail.python.org/pipermail/numpy-discussion/2018-March/077767.html


Appendix: Benchmark script
--------------------------

.. literalinclude:: nep-0016-benchmark.py


Copyright
---------

This document has been placed in the public domain.
