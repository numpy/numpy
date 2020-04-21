===================================================
NEP 37 — A dispatch protocol for NumPy-like modules
===================================================

:Author: Stephan Hoyer <shoyer@google.com>
:Author: Hameer Abbasi
:Author: Sebastian Berg
:Status: Draft
:Type: Standards Track
:Created: 2019-12-29

Abstract
--------

NEP-18's ``__array_function__`` has been a mixed success. Some projects (e.g.,
dask, CuPy, xarray, sparse, Pint, MXNet) have enthusiastically adopted it.
Others (e.g., JAX) have been more reluctant. Here we propose a new
protocol, ``__array_module__``, that we expect could eventually subsume most
use-cases for ``__array_function__``. The protocol requires explicit adoption
by both users and library authors, which ensures backwards compatibility, and
is also significantly simpler than ``__array_function__``, both of which we
expect will make it easier to adopt.

Why ``__array_function__`` hasn't been enough
---------------------------------------------

There are two broad ways in which NEP-18 has fallen short of its goals:

1. **Backwards compatibility concerns**. `__array_function__` has significant
   implications for libraries that use it:

   - `JAX <https://github.com/google/jax/issues/1565>`_ has been reluctant
     to implement ``__array_function__`` in part because it is concerned about
     breaking existing code: users expect NumPy functions like
     ``np.concatenate`` to return NumPy arrays. This is a fundamental
     limitation of the ``__array_function__`` design, which we chose to allow
     overriding the existing ``numpy`` namespace.
     Libraries like Dask and CuPy have looked at and accepted the backwards
     incompatibility impact of ``__array_function__``; it would still have been
     better for them if that impact didn't exist.

     Note that projects like `PyTorch
     <https://github.com/pytorch/pytorch/issues/22402>`_ and `scipy.sparse
     <https://github.com/scipy/scipy/issues/10362>`_ have also not
     adopted ``__array_function__`` yet, because they don't have a
     NumPy-compatible API or semantics. In the case of PyTorch, that is likely
     to be added in the future. ``scipy.sparse`` is in the same situation as
     ``numpy.matrix``: its semantics are not compatible with ``numpy.ndarray``
     and therefore adding ``__array_function__`` (except to return ``NotImplemented``
     perhaps) is not a healthy idea.
   - ``__array_function__`` currently requires an "all or nothing" approach to
     implementing NumPy's API. There is no good pathway for **incremental
     adoption**, which is particularly problematic for established projects
     for which adopting ``__array_function__`` would result in breaking
     changes.

2. **Limitations on what can be overridden.** ``__array_function__`` has some
   important gaps, most notably array creation and coercion functions:

   - **Array creation** routines (e.g., ``np.arange`` and those in
     ``np.random``) need some other mechanism for indicating what type of
     arrays to create. `NEP 35 <https://numpy.org/neps/nep-0035-array-creation-dispatch-with-array-function.html>`_
     proposed adding optional ``like=`` arguments to functions without
     existing array arguments. However, we still lack any mechanism to
     override methods on objects, such as those needed by
     ``np.random.RandomState``.
   - **Array conversion** can't reuse the existing coercion functions like
     ``np.asarray``, because ``np.asarray`` sometimes means "convert to an
     exact ``np.ndarray``" and other times means "convert to something _like_
     a NumPy array." This led to the `NEP 30
     <https://numpy.org/neps/nep-0030-duck-array-protocol.html>`_ proposal for
     a separate ``np.duckarray`` function, but this still does not resolve how
     to cast one duck array into a type matching another duck array.

Other maintainability concerns that were raised include:

- It is no longer possible to use **aliases to NumPy functions** within
  modules that support overrides. For example, both CuPy and JAX set
  ``result_type = np.result_type`` and now have to wrap use of
  ``np.result_type`` in their own ``result_type`` function instead.
- Implementing **fall-back mechanisms** for unimplemented NumPy functions
  by using NumPy's implementation is hard to get right (but see the
  `version from dask <https://github.com/dask/dask/pull/5043>`_), because
  ``__array_function__`` does not present a consistent interface.
  Converting all arguments of array type requires recursing into generic
  arguments of the form ``*args, **kwargs``.

``get_array_module`` and the ``__array_module__`` protocol
----------------------------------------------------------

We propose a new user-facing mechanism for dispatching to a duck-array
implementation, ``numpy.get_array_module``. ``get_array_module`` performs the
same type resolution as ``__array_function__`` and returns a module with an API
promised to match the standard interface of ``numpy`` that can implement
operations on all provided array types.

The protocol itself is both simpler and more powerful than
``__array_function__``, because it doesn't need to worry about actually
implementing functions. We believe it resolves most of the maintainability and
functionality limitations of ``__array_function__``.

The new protocol is opt-in, explicit and with local control; see
:ref:`appendix-design-choices` for discussion on the importance of these design
features.

The array module contract
=========================

Modules returned by ``get_array_module``/``__array_module__`` should make a
best effort to implement NumPy's core functionality on new array types(s).
Unimplemented functionality should simply be omitted (e.g., accessing an
unimplemented function should raise ``AttributeError``). In the future, we
anticipate codifying a protocol for requesting restricted subsets of ``numpy``;
see :ref:`requesting-restricted-subsets` for more details.

How to use ``get_array_module``
===============================

Code that wants to support generic duck arrays should explicitly call
``get_array_module`` to determine an appropriate array module from which to
call functions, rather than using the ``numpy`` namespace directly. For
example:

.. code:: python

    # calls the appropriate version of np.something for x and y
    module = np.get_array_module(x, y)
    module.something(x, y)

Both array creation and array conversion are supported, because dispatching is
handled by ``get_array_module`` rather than via the types of function
arguments. For example, to use random number generation functions or methods,
we can simply pull out the appropriate submodule:

.. code:: python

    def duckarray_add_random(array):
        module = np.get_array_module(array)
        noise = module.random.randn(*array.shape)
        return array + noise

We can also write the duck-array ``stack`` function from `NEP 30
<https://numpy.org/neps/nep-0030-duck-array-protocol.html>`_, without the need
for a new ``np.duckarray`` function:

.. code:: python

    def duckarray_stack(arrays):
        module = np.get_array_module(*arrays)
        arrays = [module.asarray(arr) for arr in arrays]
        shapes = {arr.shape for arr in arrays}
        if len(shapes) != 1:
            raise ValueError('all input arrays must have the same shape')
        expanded_arrays = [arr[module.newaxis, ...] for arr in arrays]
        return module.concatenate(expanded_arrays, axis=0)

By default, ``get_array_module`` will return the ``numpy`` module if no
arguments are arrays. This fall-back can be explicitly controlled by providing
the ``module`` keyword-only argument. It is also possible to indicate that an
exception should be raised instead of returning a default array module by
setting ``module=None``.

How to implement ``__array_module__``
=====================================

Libraries implementing a duck array type that want to support
``get_array_module`` need to implement the corresponding protocol,
``__array_module__``. This new protocol is based on Python's dispatch protocol
for arithmetic, and is essentially a simpler version of ``__array_function__``.

Only one argument is passed into ``__array_module__``, a Python collection of
unique array types passed into ``get_array_module``, i.e., all arguments with
an ``__array_module__`` attribute.

The special method should either return a namespace with an API matching
``numpy``, or ``NotImplemented``, indicating that it does not know how to
handle the operation:

.. code:: python

    class MyArray:
        def __array_module__(self, types):
            if not all(issubclass(t, MyArray) for t in types):
                return NotImplemented
            return my_array_module

Returning custom objects from ``__array_module__``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``my_array_module`` will typically, but need not always, be a Python module.
Returning a custom objects (e.g., with functions implemented via
``__getattr__``) may be useful for some advanced use cases.

For example, custom objects could allow for partial implementations of duck
array modules that fall-back to NumPy (although this is not recommended in
general because such fall-back behavior can be error prone):

.. code:: python

    class MyArray:
        def __array_module__(self, types):
            if all(issubclass(t, MyArray) for t in types):
                return ArrayModule()
            else:
                return NotImplemented

    class ArrayModule:
        def __getattr__(self, name):
            import base_module
            return getattr(base_module, name, getattr(numpy, name))

Subclassing from ``numpy.ndarray``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All of the same guidance about well-defined type casting hierarchies from
NEP-18 still applies. ``numpy.ndarray`` itself contains a matching
implementation of ``__array_module__``,  which is convenient for subclasses:

.. code:: python

    class ndarray:
        def __array_module__(self, types):
            if all(issubclass(t, ndarray) for t in types):
                return numpy
            else:
                return NotImplemented

NumPy's internal machinery
==========================

The type resolution rules of ``get_array_module`` follow the same model as
Python and NumPy's existing dispatch protocols: subclasses are called before
super-classes, and otherwise left to right. ``__array_module__`` is guaranteed
to be called only  a single time on each unique type.

The actual implementation of `get_array_module` will be in C, but should be
equivalent to this Python code:

.. code:: python

    def get_array_module(*arrays, default=numpy):
        implementing_arrays, types = _implementing_arrays_and_types(arrays)
        if not implementing_arrays and default is not None:
            return default
        for array in implementing_arrays:
            module = array.__array_module__(types)
            if module is not NotImplemented:
                return module
        raise TypeError("no common array module found")

    def _implementing_arrays_and_types(relevant_arrays):
        types = []
        implementing_arrays = []
        for array in relevant_arrays:
            t = type(array)
            if t not in types and hasattr(t, '__array_module__'):
                types.append(t)
                # Subclasses before superclasses, otherwise left to right
                index = len(implementing_arrays)
                for i, old_array in enumerate(implementing_arrays):
                    if issubclass(t, type(old_array)):
                        index = i
                        break
                implementing_arrays.insert(index, array)
        return implementing_arrays, types

Relationship with ``__array_ufunc__`` and ``__array_function__``
----------------------------------------------------------------

These older protocols have distinct use-cases and should remain
===============================================================

``__array_module__`` is intended to resolve limitations of
``__array_function__``, so it is natural to consider whether it could entirely
replace ``__array_function__``. This would offer dual benefits: (1) simplifying
the user-story about how to override NumPy and (2) removing the slowdown
associated with checking for dispatch when calling every NumPy function.

However, ``__array_module__`` and ``__array_function__`` are pretty different
from a user perspective: it requires explicit calls to ``get_array_function``,
rather than simply reusing original ``numpy`` functions. This is probably fine
for *libraries* that rely on duck-arrays, but may be frustratingly verbose for
interactive use.

Some of the dispatching use-cases for ``__array_ufunc__`` are also solved by
``__array_module__``, but not all of them. For example, it is still useful to
be able to define non-NumPy ufuncs (e.g., from Numba or SciPy) in a generic way
on non-NumPy arrays (e.g., with dask.array).

Given their existing adoption and distinct use cases, we don't think it makes
sense to remove or deprecate ``__array_function__`` and ``__array_ufunc__`` at
this time.

Mixin classes to implement ``__array_function__`` and ``__array_ufunc__``
=========================================================================

Despite the user-facing differences, ``__array_module__`` and a module
implementing NumPy's API still contain sufficient functionality needed to
implement dispatching with the existing duck array protocols.

For example, the following mixin classes would provide sensible defaults for
these special methods in terms of ``get_array_module`` and
``__array_module__``:

.. code:: python

    class ArrayUfuncFromModuleMixin:

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            arrays = inputs + kwargs.get('out', ())
            try:
                array_module = np.get_array_module(*arrays)
            except TypeError:
                return NotImplemented

            try:
                # Note this may have false positive matches, if ufunc.__name__
                # matches the name of a ufunc defined by NumPy. Unfortunately
                # there is no way to determine in which module a ufunc was
                # defined.
                new_ufunc = getattr(array_module, ufunc.__name__)
            except AttributeError:
                return NotImplemented

            try:
                callable = getattr(new_ufunc, method)
            except AttributeError:
                return NotImplemented

            return callable(*inputs, **kwargs)

    class ArrayFunctionFromModuleMixin:

        def __array_function__(self, func, types, args, kwargs):
            array_module = self.__array_module__(types)
            if array_module is NotImplemented:
                return NotImplemented

            # Traverse submodules to find the appropriate function
            modules = func.__module__.split('.')
            assert modules[0] == 'numpy'
            for submodule in modules[1:]:
                module = getattr(module, submodule, None)
            new_func = getattr(module, func.__name__, None)
            if new_func is None:
                return NotImplemented

            return new_func(*args, **kwargs)

To make it easier to write duck arrays, we could also add these mixin classes
into ``numpy.lib.mixins`` (but the examples above may suffice).

Alternatives considered
-----------------------

Naming
======

We like the name ``__array_module__`` because it mirrors the existing
``__array_function__`` and ``__array_ufunc__`` protocols. Another reasonable
choice could be ``__array_namespace__``.

It is less clear what the NumPy function that calls this protocol should be
called (``get_array_module`` in this proposal). Some possible alternatives:
``array_module``, ``common_array_module``, ``resolve_array_module``,
``get_namespace``, ``get_numpy``, ``get_numpylike_module``,
``get_duck_array_module``.

.. _requesting-restricted-subsets:

Requesting restricted subsets of NumPy's API
============================================

Over time, NumPy has accumulated a very large API surface, with over 600
attributes in the top level ``numpy`` module alone. It is unlikely that any
duck array library could or would want to implement all of these functions and
classes, because the frequently used subset of NumPy is much smaller.

We think it would be useful exercise to define "minimal" subset(s) of NumPy's
API, omitting rarely used or non-recommended functionality. For example,
minimal NumPy might include ``stack``, but not the other stacking functions
``column_stack``, ``dstack``, ``hstack`` and ``vstack``. This could clearly
indicate to duck array authors and users what functionality is core and what
functionality they can skip.

Support for requesting a restricted subset of NumPy's API would be a natural
feature to include in  ``get_array_function`` and ``__array_module__``, e.g.,

.. code:: python

    # array_module is only guaranteed to contain "minimal" NumPy
    array_module = np.get_array_module(*arrays, request='minimal')

To facilitate testing with NumPy and use with any valid duck array library,
NumPy itself would return restricted versions of the ``numpy`` module when
``get_array_module`` is called only on NumPy arrays. Omitted functions would
simply not exist.

Unfortunately, we have not yet figured out what these restricted subsets should
be, so it doesn't make sense to do this yet. When/if we do, we could either add
new keyword arguments to ``get_array_module`` or add new top level functions,
e.g., ``get_minimal_array_module``. We would also need to add either a new
protocol patterned off of ``__array_module__`` (e.g.,
``__array_module_minimal__``), or could add an optional second argument to
``__array_module__`` (catching errors with ``try``/``except``).

A new namespace for implicit dispatch
=====================================

Instead of supporting overrides in the main `numpy` namespace with
``__array_function__``, we could create a new opt-in namespace, e.g.,
``numpy.api``, with versions of NumPy functions that support dispatching. These
overrides would need new opt-in protocols, e.g., ``__array_function_api__``
patterned off of ``__array_function__``.

This would resolve the biggest limitations of ``__array_function__`` by being
opt-in and would also allow for unambiguously overriding functions like
``asarray``, because ``np.api.asarray`` would always mean "convert an
array-like object."  But it wouldn't solve all the dispatching needs met by
``__array_module__``, and would leave us with supporting a considerably more
complex protocol both for array users and implementors.

We could potentially implement such a new namespace *via* the
``__array_module__`` protocol. Certainly some users would find this convenient,
because it is slightly less boilerplate. But this would leave users with a
confusing choice: when should they use `get_array_module` vs.
`np.api.something`. Also, we would have to add and maintain a whole new module,
which is considerably more expensive than merely adding a function.

Dispatching on both types and arrays instead of only types
==========================================================

Instead of supporting dispatch only via unique array types, we could also
support dispatch via array objects, e.g., by passing an ``arrays`` argument as
part of the ``__array_module__`` protocol. This could potentially be useful for
dispatch for arrays with metadata, such provided by Dask and Pint, but would
impose costs in terms of type safety and complexity.

For example, a library that supports arrays on both CPUs and GPUs might decide
on which device to create a new arrays from functions like ``ones`` based on
input arguments:

.. code:: python

    class Array:
        def __array_module__(self, types, arrays):
            useful_arrays = tuple(a in arrays if isinstance(a, Array))
            if not useful_arrays:
                return NotImplemented
            prefer_gpu = any(a.prefer_gpu for a in useful_arrays)
            return ArrayModule(prefer_gpu)

    class ArrayModule:
        def __init__(self, prefer_gpu):
            self.prefer_gpu = prefer_gpu

        def __getattr__(self, name):
            import base_module
            base_func = getattr(base_module, name)
            return functools.partial(base_func, prefer_gpu=self.prefer_gpu)

This might be useful, but it's not clear if we really need it. Pint seems to
get along OK without any explicit array creation routines (favoring
multiplication by units, e.g., ``np.ones(5) * ureg.m``), and for the most part
Dask is also OK with existing ``__array_function__`` style overrides (e.g.,
favoring ``np.ones_like`` over ``np.ones``). Choosing whether to place an array
on the CPU or GPU could be solved by `making array creation lazy
<https://github.com/google/jax/pull/1668>`_.

.. _appendix-design-choices:

Appendix: design choices for API overrides
------------------------------------------

There is a large range of possible design choices for overriding NumPy's API.
Here we discuss three major axes of the design decision that guided our design
for ``__array_module__``.

Opt-in vs. opt-out for users
============================

The ``__array_ufunc__`` and ``__array_function__`` protocols provide a
mechanism for overriding NumPy functions *within NumPy's existing namespace*.
This means that users need to explicitly opt-out if they do not want any
overridden behavior, e.g., by casting arrays with ``np.asarray()``.

In theory, this approach lowers the barrier for adopting these protocols in
user code and libraries, because code that uses the standard NumPy namespace is
automatically compatible. But in practice, this hasn't worked out. For example,
most well-maintained libraries that use NumPy follow the best practice of
casting all inputs with ``np.asarray()``, which they would have to explicitly
relax to use ``__array_function__``. Our experience has been that making a
library compatible with a new duck array type typically requires at least a
small amount of work to accommodate differences in the data model and operations
that can be implemented efficiently.

These opt-out approaches also considerably complicate backwards compatibility
for libraries that adopt these protocols, because by opting in as a library
they also opt-in their users, whether they expect it or not. For winning over
libraries that have been unable to adopt ``__array_function__``, an opt-in
approach seems like a must.

Explicit vs. implicit choice of implementation
==============================================

Both ``__array_ufunc__`` and ``__array_function__`` have implicit control over
dispatching: the dispatched functions are determined via the appropriate
protocols in every function call. This generalizes well to handling many
different types of objects, as evidenced by its use for implementing arithmetic
operators in Python, but it has an important downside for **readability**:
it is not longer immediately evident to readers of code what happens when a
function is called, because the function's implementation could be overridden
by any of its arguments.

The **speed** implications are:

- When using a *duck-array type*, ``get_array_module`` means type checking only
  needs to happen once inside each function that supports duck typing, whereas
  with ``__array_function__`` it happens every time a NumPy function is called.
  Obvious it's going to depend on the function, but if a typical duck-array
  supporting function calls into other NumPy functions 3-5 times this is a factor
  of 3-5x more overhead.
- When using *NumPy arrays*, ``get_array_module`` is one extra call per
  function (``__array_function__`` overhead remains the same), which means a
  small amount of extra overhead.

Explicit and implicit choice of implementations are not mutually exclusive
options. Indeed, most implementations of NumPy API overrides via
``__array_function__`` that we are familiar with (namely, Dask, CuPy and
Sparse, but not Pint) also include an explicit way to use their version of
NumPy's API by importing a module directly (``dask.array``, ``cupy`` or
``sparse``, respectively).

Local vs. non-local vs. global control
======================================

The final design axis is how users control the choice of API:

- **Local control**, as exemplified by multiple dispatch and Python protocols for
  arithmetic, determines which implementation to use either by checking types
  or calling methods on the direct arguments of a function.
- **Non-local control** such as `np.errstate
  <https://docs.scipy.org/doc/numpy/reference/generated/numpy.errstate.html>`_
  overrides behavior with global-state via function decorators or
  context-managers. Control is determined hierarchically, via the inner-most
  context.
- **Global control** provides a mechanism for users to set default behavior,
  either via function calls or configuration files. For example, matplotlib
  allows setting a global choice of plotting backend.

Local control is generally considered a best practice for API design, because
control flow is entirely explicit, which makes it the easiest to understand.
Non-local and global control are occasionally used, but generally either due to
ignorance or a lack of better alternatives.

In the case of duck typing for NumPy's public API, we think non-local or global
control would be mistakes, mostly because they **don't compose well**. If one
library sets/needs one set of overrides and then internally calls a routine
that expects another set of overrides, the resulting behavior may be very
surprising. Higher order functions are especially problematic, because the
context in which functions are evaluated may not be the context in which they
are defined.

One class of override use cases where we think non-local and global control are
appropriate is for choosing a backend system that is guaranteed to have an
entirely consistent interface, such as a faster alternative implementation of
``numpy.fft`` on NumPy arrays. However, these are out of scope for the current
proposal, which is focused on duck arrays.
