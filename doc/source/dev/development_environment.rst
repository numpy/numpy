.. _development-environment:

Setting up and using your development environment
=================================================

.. _recommended-development-setup:

Recommended development setup
-----------------------------

Since NumPy contains parts written in C and Cython that need to be
compiled before use, make sure you have the necessary compilers and Python
development headers installed - see :ref:`building-from-source`. Building
NumPy as of version ``2.0`` requires C11 and C++17 compliant compilers.

Having compiled code also means that importing NumPy from the development
sources needs some additional steps, which are explained below.  For the rest
of this chapter we assume that you have set up your git repo as described in
:ref:`using-git`.

.. note::

   If you are having trouble building NumPy from source or setting up your
   local development environment, you can try to build NumPy with GitHub
   Codespaces. It allows you to create the correct development environment
   right in your browser, reducing the need to install local development
   environments and deal with incompatible dependencies.

   If you have good internet connectivity and want a temporary set-up, it is
   often faster to work on NumPy in a Codespaces environment. For documentation
   on how to get started with Codespaces, see
   `the Codespaces docs <https://docs.github.com/en/codespaces>`__.
   When creating a codespace for the ``numpy/numpy`` repository, the default
   2-core machine type works; 4-core will build and work a bit faster (but of
   course at a cost of halving your number of free usage hours). Once your
   codespace has started, you can run ``conda activate numpy-dev`` and your
   development environment is completely set up - you can then follow the
   relevant parts of the NumPy documentation to build, test, develop, write
   docs, and contribute to NumPy.

Using virtual environments
--------------------------

A frequently asked question is "How do I set up a development version of NumPy
in parallel to a released version that I use to do my job/research?".

One simple way to achieve this is to install the released version in
site-packages, by using pip or conda for example, and set
up the development version in a virtual environment.

If you use conda, we recommend creating a separate virtual environment for
numpy development using the ``environment.yml`` file in the root of the repo
(this will create the environment and install all development dependencies at
once)::

    $ conda env create -f environment.yml  # `mamba` works too for this command
    $ conda activate numpy-dev

If you installed Python some other way than conda, first install
`virtualenv`_ (optionally use `virtualenvwrapper`_), then create your
virtualenv (named ``numpy-dev`` here), activate it, and install all project 
dependencies with::

    $ virtualenv numpy-dev
    $ source numpy-dev/bin/activate # activate virtual environment
    $ python -m pip install -r requirements/all_requirements.txt

Now, whenever you want to switch to the virtual environment, you can use the
command ``source numpy-dev/bin/activate``, and ``deactivate`` to exit from the
virtual environment and back to your previous shell.

Building from source
--------------------

See :ref:`building-from-source`.

.. _testing-builds:

Testing builds
--------------

Before running the tests, first install the test dependencies::

    $ python -m pip install -r requirements/test_requirements.txt
    $ python -m pip install asv # only for running benchmarks

To build the development version of NumPy and run tests, spawn
interactive shells with the Python import paths properly set up etc., use the
`spin <https://github.com/scientific-python/spin>`_ utility. To run tests, do
one of::

    $ spin test -v
    $ spin test numpy/random  # to run the tests in a specific module
    $ spin test -v -t numpy/_core/tests/test_nditer.py::test_iter_c_order

This builds NumPy first, so the first time it may take a few minutes.

You can also use ``spin bench`` for benchmarking. See ``spin --help`` for more
command line options.

.. note::

    If the above commands result in ``RuntimeError: Cannot parse version 0+untagged.xxxxx``,
    run ``git pull upstream main --tags``.

Additional arguments may be forwarded to ``pytest`` by passing the extra
arguments after a bare ``--``. For example, to run a test method with the
``--pdb`` flag forwarded to the target, run the following::

    $ spin test -t numpy/tests/test_scripts.py::test_f2py -- --pdb

You can also  `match test names using python operators`_ by passing the ``-k``
argument to pytest::

    $ spin test -v -t numpy/_core/tests/test_multiarray.py -- -k "MatMul and not vector"

To run "doctests" -- to check that the code examples in the documentation is correct --
use the `check-docs` spin command. It relies on the `scipy-docs` package, which
provides several additional features on top of the standard library ``doctest``
package. Install ``scipy-doctest`` and run one of::

  $ spin check-docs -v
  $ spin check-docs numpy/linalg
  $ spin check-docs -v -- -k 'det and not slogdet'

.. note::

    Remember that all tests of NumPy should pass before committing your changes.

.. note::

   Some of the tests in the test suite require a large amount of
   memory, and are skipped if your system does not have enough.

..
   To override the automatic detection of available memory, set the
   environment variable ``NPY_AVAILABLE_MEM``, for example
   ``NPY_AVAILABLE_MEM=32GB``, or using pytest ``--available-memory=32GB``
   target option.

Other build options
-------------------

For more options including selecting compilers, setting custom compiler flags
and controlling parallelism, see :doc:`scipy:building/compilers_and_options`
(from the SciPy documentation.)


Running tests
-------------

Besides using ``spin``, there are various ways to run the tests.  Inside
the interpreter, tests can be run like this::

    >>> np.test()  # doctest: +SKIPBLOCK
    >>> np.test('full')   # Also run tests marked as slow
    >>> np.test('full', verbose=2)   # Additionally print test name/file

    An example of a successful test :
    ``4686 passed, 362 skipped, 9 xfailed, 5 warnings in 213.99 seconds``

Or a similar way from the command line::

    $ python -c "import numpy as np; np.test()"

Tests can also be run with ``pytest numpy``, however then the NumPy-specific
plugin is not found which causes strange side effects.

Running individual test files can be useful; it's much faster than running the
whole test suite or that of a whole module (example: ``np.random.test()``).
This can be done with::

    $ python path_to_testfile/test_file.py

That also takes extra arguments, like ``--pdb`` which drops you into the Python
debugger when a test fails or an exception is raised.

Running tests with `tox`_ is also supported.  For example, to build NumPy and
run the test suite with Python 3.9, use::

    $ tox -e py39

For more extensive information, see :ref:`testing-guidelines`.

Note: do not run the tests from the root directory of your numpy git repo without ``spin``,
that will result in strange test errors.

Running type checks
-------------------
Changes that involve static type declarations are also executed using ``spin``.
The invocation will look like the following:

    $ spin mypy

This will look in the ``typing/tests`` directory for sets of operations to
test for type incompatibility.

Running linting
---------------
Lint checks can be performed on newly added lines of Python code.

Install all dependent packages using pip::

    $ python -m pip install -r requirements/linter_requirements.txt

To run lint checks before committing new code, run::

    $ python tools/linter.py

To check all changes in newly added Python code of current branch with target branch, run::

    $ python tools/linter.py

If there are no errors, the script exits with no message. In case of errors,
check the error message for details::

    $ python tools/linter.py
    ./numpy/_core/tests/test_scalarmath.py:34:5: E303 too many blank lines (3)
    1       E303 too many blank lines (3)

It is advisable to run lint checks before pushing commits to a remote branch
since the linter runs as part of the CI pipeline.

For more details on Style Guidelines:

- `Python Style Guide`_
- :ref:`NEP45`

Rebuilding & cleaning the workspace
-----------------------------------

Rebuilding NumPy after making changes to compiled code can be done with the
same build command as you used previously - only the changed files will be
re-built.  Doing a full build, which sometimes is necessary, requires cleaning
the workspace first.  The standard way of doing this is (*note: deletes any
uncommitted files!*)::

    $ git clean -xdf

When you want to discard all changes and go back to the last commit in the
repo, use one of::

    $ git checkout .
    $ git reset --hard


.. _debugging:

Debugging
---------

Another frequently asked question is "How do I debug C code inside NumPy?".
First, ensure that you have gdb installed on your system with the Python
extensions (often the default on Linux). You can see which version of
Python is running inside gdb to verify your setup::

    (gdb) python
    >import sys
    >print(sys.version_info)
    >end
    sys.version_info(major=3, minor=7, micro=0, releaselevel='final', serial=0)

Most python builds do not include debug symbols and are built with compiler
optimizations enabled. To get the best debugging experience using a debug build
of Python is encouraged, see :ref:`advanced_debugging`.

In terms of debugging, NumPy also needs to be built in a debug mode. You need to use
``debug`` build type and disable optimizations to make sure ``-O0`` flag is used
during object building. Note that NumPy should NOT be installed in your environment
before you build with the ``spin build`` command.

To generate source-level debug information within the build process run::

    $ spin build --clean -- -Dbuildtype=debug -Ddisable-optimization=true

.. note::

    In case you are using conda environment be aware that conda sets ``CFLAGS``
    and ``CXXFLAGS`` automatically, and they will include the ``-O2`` flag by default.
    You can safely use ``unset CFLAGS && unset CXXFLAGS`` to avoid them or provide them
    at the beginning of the ``spin`` command: ``CFLAGS="-O0 -g" CXXFLAGS="-O0 -g"``.
    Alternatively, to take control of these variables more permanently, you can create
    ``env_vars.sh`` file in the ``<path-to-conda-envs>/numpy-dev/etc/conda/activate.d``
    directory. In this file you can export ``CFLAGS`` and ``CXXFLAGS`` variables.
    For complete instructions please refer to
    https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#saving-environment-variables.


Next you need to write a Python script that invokes the C code whose execution
you want to debug. For instance ``mytest.py``::

    import numpy as np
    x = np.arange(5)
    np.empty_like(x)

Note that your test file needs to be outside the NumPy clone you have. Now, you can
run::

    $ spin gdb /path/to/mytest.py

In case you are using clang toolchain::

    $ spin lldb /path/to/mytest.py

And then in the debugger::

    (gdb) break array_empty_like
    (gdb) run

lldb counterpart::

    (lldb) breakpoint set --name array_empty_like
    (lldb) run

The execution will now stop at the corresponding C function and you can step
through it as usual. A number of useful Python-specific commands are available.
For example to see where in the Python code you are, use ``py-list``, to see the
python traceback, use ``py-bt``.  For more details, see
`DebuggingWithGdb`_. Here are some commonly used commands:

- ``list``: List specified function or line.
- ``next``: Step program, proceeding through subroutine calls.
- ``step``: Continue program being debugged, after signal or breakpoint.
- ``print``: Print value of expression EXP.

Rich support for Python debugging requires that the ``python-gdb.py`` script
distributed with Python is installed in a path where gdb can find it. If you
installed your Python build from your system package manager, you likely do
not need to manually do anything. However, if you built Python from source,
you will likely need to create a ``.gdbinit`` file in your home directory
pointing gdb at the location of your Python installation. For example, a
version of python installed via `pyenv <https://github.com/pyenv/pyenv>`_
needs a ``.gdbinit`` file with the following contents:

.. code-block:: text

    add-auto-load-safe-path ~/.pyenv

Building NumPy with a Python built with debug support (on Linux distributions
typically packaged as ``python-dbg``) is highly recommended.

.. _DebuggingWithGdb: https://wiki.python.org/moin/DebuggingWithGdb
.. _tox: https://tox.readthedocs.io/
.. _virtualenv: https://virtualenv.pypa.io/
.. _virtualenvwrapper: https://doughellmann.com/projects/virtualenvwrapper/
.. _Waf: https://code.google.com/p/waf/
.. _`match test names using python operators`: https://docs.pytest.org/en/latest/usage.html#specifying-tests-selecting-tests
.. _`Python Style Guide`: https://www.python.org/dev/peps/pep-0008/

Understanding the code & getting started
----------------------------------------

The best strategy to better understand the code base is to pick something you
want to change and start reading the code to figure out how it works. When in
doubt, you can ask questions on the mailing list. It is perfectly okay if your
pull requests aren't perfect, the community is always happy to help. As a
volunteer project, things do sometimes get dropped and it's totally fine to
ping us if something has sat without a response for about two to four weeks.

So go ahead and pick something that annoys or confuses you about NumPy,
experiment with the code, hang around for discussions or go through the
reference documents to try to fix it. Things will fall in place and soon
you'll have a pretty good understanding of the project as a whole. Good Luck!
