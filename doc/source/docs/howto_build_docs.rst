.. _howto-build-docs:

=========================================
Building the NumPy API and reference docs
=========================================

We currently use Sphinx_ for generating the API and reference
documentation for NumPy.  You will need Sphinx >= 2.2.0.

If you only want to get the documentation, note that pre-built
versions can be found at

    https://numpy.org/doc/

in several different formats.

.. _Sphinx: http://www.sphinx-doc.org/

To build the documentation, you must first build NumPy.

There are two options for building NumPy and its documentation- building with
Gitpod or locally from source. Your choice depends on your operating system and
familiarity with the command line.

Gitpod
------------

Gitpod is an open-source platform that automatically creates
the correct development environment right in your browser, reducing the need to
install local development environments and deal with incompatible dependencies.

If you are a Windows user, unfamiliar with using the command line or building
NumPy's documentation for the first time, it is often faster to build with
Gitpod. Here are the in-depth instructions for building NumPy with `building
NumPy with Gitpod`_.

.. _building NumPy with Gitpod: https://numpy.org/devdocs/dev/development_gitpod.html

Building locally
------------------

Building locally on your machine gives you more granular control. If you are a
MacOS or Linux user familiar with using the command line, you can continue with
building NumPy locally by following the instructions below.


Instructions
------------

If you obtained NumPy via git, get also the git submodules that contain
additional parts required for building the documentation::

    git submodule update --init

In addition, building the documentation requires the Sphinx extension
`plot_directive`, which is shipped with Matplotlib_. This Sphinx extension can
be installed by installing Matplotlib. You will also need Python>=3.6.

Since large parts of the main documentation are obtained from numpy via
``import numpy`` and examining the docstrings, you will need to first build
NumPy, and install it so that the correct version is imported.

After NumPy is installed, install SciPy since some of the plots in the random
module require `scipy.special` to display properly.

Note that you can eg. install NumPy to a temporary location and set
the PYTHONPATH environment variable appropriately.
Alternatively, if using Python virtual environments (via e.g. ``conda``,
``virtualenv`` or the ``venv`` module), installing numpy into a
new virtual environment is recommended.
All of the necessary dependencies for building the NumPy docs can be installed
with::

    pip install -r doc_requirements.txt

Now you are ready to generate the docs, so write::

    cd doc
    make html

in the ``doc/`` directory. If all goes well, this will generate a
``build/html`` subdirectory containing the built documentation. If you get
a message about ``installed numpy != current repo git version``, you must
either override the check by setting ``GITVER`` or re-install NumPy.

If you have built numpy into a virtual environment and get an error
that says ``numpy not found, cannot build documentation without...``,
you need to override the makefile ``PYTHON`` variable at the command
line, so instead of writing ``make  html`` write::

    make PYTHON=python html
    

Note that building the documentation on Windows is currently not actively
supported, though it should be possible. (See Sphinx_ documentation
for more information.)

To build the PDF documentation, do instead::

   make latex
   make -C build/latex all-pdf

You will need to have Latex installed for this, inclusive of support for
Greek letters.  For example, on Ubuntu xenial ``texlive-lang-greek`` and
``cm-super`` are needed.  Also ``latexmk`` is needed on non-Windows systems.

Instead of the above, you can also do::

   make dist

which will rebuild NumPy, install it to a temporary location, and
build the documentation in all formats. This will most likely again
only work on Unix platforms.

The documentation for NumPy distributed at https://numpy.org/doc in html and
pdf format is also built with ``make dist``.  See `HOWTO RELEASE`_ for details
on how to update https://numpy.org/doc.

.. _Matplotlib: https://matplotlib.org/
.. _HOWTO RELEASE: https://github.com/numpy/numpy/blob/main/doc/HOWTO_RELEASE.rst.txt

Sphinx extensions
-----------------

NumPy's documentation uses several custom extensions to Sphinx.  These
are shipped in the ``sphinxext/`` directory (as git submodules, as discussed
above), and are automatically enabled when building NumPy's documentation.

If you want to make use of these extensions in third-party
projects, they are available on PyPi_ as the numpydoc_ package.

.. _PyPi: https://pypi.org/
.. _numpydoc: https://python.org/pypi/numpydoc
