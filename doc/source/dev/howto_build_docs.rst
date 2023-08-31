.. _howto-build-docs:

=========================================
Building the NumPy API and reference docs
=========================================

If you only want to get the documentation, note that pre-built
versions can be found at

https://numpy.org/doc/

in several different formats.

Development environments
========================

Before proceeding further it should be noted that the documentation is built
with the ``make`` tool, which is not natively available on Windows. MacOS or
Linux users can jump to :ref:`how-todoc.prerequisites`. It is recommended for
Windows users to set up their development environment on
GitHub Codespaces (see :ref:`recommended-development-setup`) or
`Windows Subsystem for Linux (WSL) <https://learn.microsoft.com/en-us/windows/wsl/install>`_.
WSL is a good option for a persistent local set-up.


.. _how-todoc.prerequisites:

Prerequisites
=============

Building the NumPy documentation and API reference requires the following:

NumPy
~~~~~

Since large parts of the main documentation are obtained from NumPy via
``import numpy`` and examining the docstrings, you will need to first
:ref:`build <development-environment>` and install it so that the correct version is
imported.
NumPy has to be re-built and re-installed every time you fetch the latest version of the
repository, before generating the documentation. This ensures that the NumPy version and
the git repository version are in sync.

Note that you can e.g. install NumPy to a temporary location and set
the PYTHONPATH environment variable appropriately.
Alternatively, if using Python virtual environments (via e.g. ``conda``,
``virtualenv`` or the ``venv`` module), installing NumPy into a
new virtual environment is recommended.

Dependencies
~~~~~~~~~~~~

All of the necessary dependencies for building the NumPy docs except for
Doxygen_ can be installed with::

    pip install -r doc_requirements.txt

We currently use Sphinx_ along with Doxygen_ for generating the API and
reference documentation for NumPy. In addition, building the documentation
requires the Sphinx extension `plot_directive`, which is shipped with
:doc:`Matplotlib <matplotlib:index>`. We also use numpydoc_ to render docstrings in
the generated API documentation. :doc:`SciPy <scipy:index>`
is installed since some parts of the documentation require SciPy functions.

For installing Doxygen_, please check the official
`download <https://www.doxygen.nl/download.html#srcbin>`_ and
`installation <https://www.doxygen.nl/manual/install.html>`_ pages, or if you
are using Linux then you can install it through your distribution package manager.

.. note::

   Try to install a newer version of Doxygen_ > 1.8.10 otherwise you may get some
   warnings during the build.

Submodules
~~~~~~~~~~

If you obtained NumPy via git, also get the git submodules that contain
additional parts required for building the documentation::

    git submodule update --init

.. _Sphinx: http://www.sphinx-doc.org/
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/index.html
.. _Doxygen: https://www.doxygen.nl/index.html

Instructions
============

Now you are ready to generate the docs, so write::

    spin docs

This will build NumPy from source if you haven't already, and run Sphinx to
build the ``html`` docs. If all goes well, this will generate a ``build/html``
subdirectory in the ``/doc`` directory, containing the built documentation.

The documentation for NumPy distributed at https://numpy.org/doc in html and
pdf format is also built with ``make dist``.  See `HOWTO RELEASE`_ for details
on how to update https://numpy.org/doc.

.. _LaTeX: https://www.latex-project.org/
.. _HOWTO RELEASE: https://github.com/numpy/numpy/blob/main/doc/HOWTO_RELEASE.rst
