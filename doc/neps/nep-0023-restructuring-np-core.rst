==============================================
NEP 23 — Making details of ``np.core`` private
==============================================

:Author: Eric Wieser <wieser.eric+numpy@gmail.com>
:Status: Draft
:Type: Standards Track
:Created: 2018-09-09
:Resolution: <url> (required for Accepted | Rejected | Withdrawn)

Abstract
--------

The ``np.core`` package consists of a collection of submodules, most of which
have a name not prefixed with an underscore - the pythonic way to indicate a
public API.  The naive user will consider the full set of members within any of
these modules as a public API. Since these submodules import names from other
submodules, these imported names appear to be part of the public API too.

As a result of this, any of the following can be seen as a breaking change:

* Moving a function from one file to another
* Removing an "unused" import from a file

This traps us into creating increasingly convoluted [1]_ and cyclic imports,
under the constant fear that a reorganization will break downstream code.

This NEP does not consider ``np.lib``.

Detailed description
--------------------

The list of existing modules within ``np.core`` is::

	>>> import inspect
	>>> [name for name, mod in inspect.getmembers(np.core, inspect.ismodule)]
	['_internal',
	 '_methods',
	 'arrayprint',
	 'char',
	 'defchararray',
	 'einsumfunc',
	 'fromnumeric',
	 'function_base',
	 'getlimits',
	 'info',
	 'machar',
	 'multiarray',
	 'numeric',
	 'numerictypes',
	 'rec',
	 'records',
	 'shape_base',
	 'umath']

The publicness of this API surface is contagious - in principle,
``np.core.numerictypes.array`` (an alias of ``np.core.multiarray.array``) is
part of the API too.

This is problematic - it means that if ``numeric.py`` contains ``from
numpy.core.multiarray import array``, then we can never remove that line
without risking breaking someone using ``np.core.numeric.array``.

The proposal here is to:

* Make all of the existing modules within `np.core` private, adding a leading
  underscore to their names.
* Introduce a set of new modules with the same names as the old ones, which:

  * hard-code a list of the public API surface of their module at the time of
    its creation
  * emit a ``DeprecationWarning`` upon import, warning the user that they are
    relying on implementation details.


Implementation
--------------

1. Rename every module ``numpy/core/<module>.py`` to  ``numpy/core/_<module>.py``
2. Change ``from <module> import ...`` in ``numpy/core/__init__.py`` to
   ``from _<module> import ...``, and similarly for other imports.
3. Create a new module for each existing one. For instance,
   ``numpy/core/numeric.py`` would look like::

    """
    This module exists to ensure compatibility of our public API. This is a
    snapshot of our unintentionally public API as taken to implement NEP23.
    """
    import warnings
    warnings.warn(
    	"np.core.numeric is an implementation detail, and may be removed in future. If you are using a member of this module, you should be importing it from np.core directly.", DeprecationWarning
    )

    _globals = set(globals())

    # the value of ``np.core.numeric.__all__`` in the pre-NEP23 revision
    __all__ = [
		'ALLOW_THREADS',
		'AxisError',
		'BUFSIZE',
		'CLIP',
		...
    ]


    # the public values in set(dir(np.core.numeric)) - set(__all__) in the pre-NEP23 revision
    _old_dir = [
		'absolute_import',
		'basestring',
		'builtins',
		'collections',
		'division',
		'extend_all',
		'fromnumeric',
		...
    ]

    # the exact modules these are imported from is now free to change
    from ._numeric import (
    	ALLOW_THREADS,
    	AxisError
    )

    # verify that we did not expose anything we should not have
    _exposed = set(globals()) - _globals
    assert _exposed == set(__all__) | set(_old_dir)


Something to be aware of here will be the fact that ``dir`` of
``np.core.numerictypes``, and of any module that does ``from
np.core.numerictypes import *`` is platform-dependent, as it contains all the
type aliases.

Backward compatibility
----------------------

In some cases, there may be members at ``np.core.*.*`` which we intended to be
somewhat public-facing.  An example of such a function is
`numpy.core.numeric.normalize_axis_index`, which downstream libraries are
starting to use to validate axis arguments.  Users of these functions will
receive ``DeprecationWarning``s until we lift these members to ``np.core.*``.

Alternatives
------------

* Declare our public API is determined solely by `__all__`, and that users
  relying on members not included there are on their own
* Declare that only ``np.core.*`` is public API, and ``np.core.*.*`` is private

Discussion
----------

TODO

References and Footnotes
------------------------

.. [1] `Where can I find a simple description of the delineation of NumPy's modules?
       <https://github.com/numpy/numpy/issues/11513>`_


Copyright
---------

This document has been placed in the public domain.
