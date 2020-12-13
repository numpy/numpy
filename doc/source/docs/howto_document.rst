.. _howto-document:


A Guide to NumPy Documentation
==============================

.. _userdoc_guide:

User documentation
******************
- In general, we follow the
  `Google developer documentation style guide <https://developers.google.com/style>`_.

- NumPy style governs cases where:

      - Google has no guidance, or
      - We prefer not to use the Google style

  Our current rules:

      - We pluralize *index* as *indices* rather than
        `indexes <https://developers.google.com/style/word-list#letter-i>`_,
        following the precedent of :func:`numpy.indices`.

      - For consistency we also pluralize *matrix* as *matrices*.

- Grammatical issues inadequately addressed by the NumPy or Google rules are
  decided by the section on "Grammar and Usage" in the most recent edition of
  the `Chicago Manual of Style
  <https://en.wikipedia.org/wiki/The_Chicago_Manual_of_Style>`_.

- We welcome being
  `alerted <https://github.com/numpy/numpy/issues>`_ to cases
  we should add to the NumPy style rules.



.. _docstring_intro:

Docstrings
**********

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

Please use the numpydoc `formatting standard`_ as shown in their example_

.. _`formatting standard`: https://numpydoc.readthedocs.io/en/latest/format.html
.. _example: https://numpydoc.readthedocs.io/en/latest/example.html
