.. _howto-document:


A Guide to NumPy/SciPy Documentation
====================================

When using `Sphinx <http://www.sphinx-doc.org/>`__ in combination with the
numpy conventions, you should use the ``numpydoc`` extension so that your
docstrings will be handled correctly. For example, Sphinx will extract the
``Parameters`` section from your docstring and convert it into a field
list.  Using ``numpydoc`` will also avoid the reStructuredText errors produced
by plain Sphinx when it encounters numpy docstring conventions like
section headers (e.g. ``-------------``) that sphinx does not expect to
find in docstrings.

Some features described in this document require a recent version of
``numpydoc``. For example, the **Yields** section was added in
``numpydoc`` 0.6.

It is available from:

* `numpydoc on PyPI <https://pypi.python.org/pypi/numpydoc>`_
* `numpydoc on GitHub <https://github.com/numpy/numpydoc/>`_

Note that for documentation within numpy, it is not necessary to do
``import numpy as np`` at the beginning of an example.  However, some
sub-modules, such as ``fft``, are not imported by default, and you have to
include them explicitly::

  import numpy.fft

after which you may use it::

  np.fft.fft2(...)

.. rubric::
    **For convenience the** `formatting standard`_ **is included below with an
    example**

.. include:: ../../sphinxext/doc/format.rst

.. _example:

Example Source
==============

.. literalinclude:: ../../sphinxext/doc/example.py



Example Rendered
================

.. ifconfig:: python_version_major < '3'

    The example is rendered only when sphinx is run with python3 and above

.. automodule:: doc.example
    :members:

.. _`formatting standard`: https://numpydoc.readthedocs.io/en/latest/format.html
