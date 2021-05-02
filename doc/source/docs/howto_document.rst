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

.. _doc_c_code:

Documenting C/C++ Code
**********************

Recently, NumPy headed out to use Doxygen_ along with Sphinx due to the urgent need
to generate C/C++ API reference documentation from comment blocks, especially with
the latter's expansion in the use of SIMD, and also due to the future reliance on C++.
Thanks to Breathe_, we were able to bind both worlds together at the lowest cost.

It takes three steps to complete the documentation process:

Writing the comment blocks
--------------------------

So far, there is no set commenting style yet to follow, but the Javadoc style
is more preferred than other styles due to the similarities with the current
existing non-indexed comment blocks.

.. note::
   If you have never used Doxygen_ before, then maybe you need to take a quick
   look at `"Documenting the code" <https://www.doxygen.nl/manual/docblocks.html>`_.

This is how Javadoc style looks like::

  /**
   * Your brief is here.
   */
  void a_c_function(void);

Doxygen_'s `JAVADOC_AUTOBRIEF <https://www.doxygen.nl/manual/config.html#cfg_javadoc_autobrief>`_
is enabled in the configuration file which means the first sentence of the documentation block is
automatically treated as a brief description, in other words no need for
`@brief <https://www.doxygen.nl/manual/commands.html#cmdbrief>`_ command.

reSt markup is fully supported, but the use of reST requires it to be included
in the `@rst` command, like this:

.. literalinclude:: examples/c_doxygen_rst.h

.. doxygenfunction:: rst_example1_c_function

Feeding Doxygen
---------------

Not all headers files are colletcted automatically. you will have to add the desired
C/C++ header paths within the sub-config files of Doxygen.

Sub-config files have the unique name `.doxyfile`; you can usually find them near
directories that contain documented headers. You will have to create a new config file if
there's no one located in a path close(2-depth) to the headers you want to add.

Sub-config files can accept any of Doxygen_ `configuration options <https://www.doxygen.nl/manual/config.html>`_,
but be aware of override or re-initialize any configuration option,
you must only use the concatenation operator `+=`. for example ::

   # to specfiy certain headers
   INPUT += @CUR_DIR/header1.h \
            @CUR_DIR/header2.h
   # to add all headers in certain path
   INPUT += @CUR_DIR/to/headers
   # to define certain macros
   PREDEFINED += C_MACRO(X)=X
   # to enable certain branches
   PREDEFINED += NPY_HAVE_FEATURE \
                 NPY_HAVE_FEATURE2

.. note::

    @CUR_DIR is a template constant returns the current
    dir path of the sub-config file.

Inclusion directives
--------------------

Breathe_ provides a wide range collection of custom directives to allow
combining the generated documents by Doxygen_.

.. note::

   For more information, please check out "`Directives & Config Variables <https://breathe.readthedocs.io/en/latest/directives.html>`_"


Assume having a C header contains a group of functions, and types:

.. literalinclude:: examples/c_doxygen_group.h
.. doxygengroup:: C_doxygen_group_example
    :members:
    :undoc-members:

.. _`Doxygen`: https://www.doxygen.nl/index.html
.. _`Breathe`: https://breathe.readthedocs.io/en/latest/
