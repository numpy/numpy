.. _howto-build-docs:

=========================================
Building the NumPy API and reference docs
=========================================

If you only want to get the documentation, note that pre-built
versions can be found at

    https://numpy.org/doc/

in several different formats.

Development environments
------------------------

Before proceeding further it should be noted that the documentation is built with the ``make`` tool,
which is not natively available on Windows. MacOS or Linux users can jump
to :ref:`how-todoc.prerequisites`. It is recommended for Windows users to set up their development
environment on :ref:`Gitpod <development-gitpod>` or `Windows Subsystem
for Linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_. WSL is a good option
for a persistent local set-up.

Gitpod
^^^^^^
Gitpod is an open-source platform that automatically creates the correct development environment right
in your browser, reducing the need to install local development environments and deal with
incompatible dependencies.

If you have good internet connectivity and want a temporary set-up,
it is often faster to build with Gitpod. Here are the in-depth instructions for
:ref:`building NumPy with Gitpod <development-gitpod>`.


.. _how-todoc.prerequisites:

Prerequisites
-------------

Building the NumPy documentation and API reference requires the following:

NumPy
^^^^^

Since large parts of the main documentation are obtained from NumPy via
``import numpy`` and examining the docstrings, you will need to first
:ref:`build <building-from-source>` and install it so that the correct version is imported.
NumPy has to be re-built and re-installed every time you fetch the latest version of the
repository, before generating the documentation. This ensures that the NumPy version and
the git repository version are in sync.

Note that you can e.g. install NumPy to a temporary location and set
the PYTHONPATH environment variable appropriately.
Alternatively, if using Python virtual environments (via e.g. ``conda``,
``virtualenv`` or the ``venv`` module), installing NumPy into a
new virtual environment is recommended.

Dependencies
^^^^^^^^^^^^

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
^^^^^^^^^^

If you obtained NumPy via git, also get the git submodules that contain
additional parts required for building the documentation::

    git submodule update --init

.. _Sphinx: http://www.sphinx-doc.org/
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/index.html
.. _Doxygen: https://www.doxygen.nl/index.html

Instructions
------------

Now you are ready to generate the docs, so write::

    cd doc
    make html

If all goes well, this will generate a
``build/html`` subdirectory in the ``/doc`` directory, containing the built documentation. If
you get a message about ``installed numpy != current repo git version``, you must
either override the check by setting ``GITVER`` or re-install NumPy.

If you have built NumPy into a virtual environment and get an error
that says ``numpy not found, cannot build documentation without...``,
you need to override the makefile ``PYTHON`` variable at the command
line, so instead of writing ``make  html`` write::

    make PYTHON=python html

To build the PDF documentation, do instead::

   make latex
   make -C build/latex all-pdf

You will need to have LaTeX_ installed for this, inclusive of support for
Greek letters.  For example, on Ubuntu xenial ``texlive-lang-greek`` and
``cm-super`` are needed.  Also, ``latexmk`` is needed on non-Windows systems.

Instead of the above, you can also do::

   make dist

which will rebuild NumPy, install it to a temporary location, and
build the documentation in all formats. This will most likely again
only work on Unix platforms.

The documentation for NumPy distributed at https://numpy.org/doc in html and
pdf format is also built with ``make dist``.  See `HOWTO RELEASE`_ for details
on how to update https://numpy.org/doc.

.. _LaTeX: https://www.latex-project.org/
.. _HOWTO RELEASE: https://github.com/numpy/numpy/blob/main/doc/HOWTO_RELEASE.rst.txt
