.. _development-environment:

Setting up and using your development environment
=================================================


Recommended development setup
-----------------------------

Since NumPy contains parts written in C and Cython that need to be
compiled before use, make sure you have the necessary compilers and Python
development headers installed - see :ref:`building-from-source`.

Having compiled code also means that importing Numpy from the development
sources needs some additional steps, which are explained below.  For the rest
of this chapter we assume that you have set up your git repo as described in
:ref:`using-git`.

To build the development version of NumPy and run tests, spawn
interactive shells with the Python import paths properly set up etc.,
do one of::

    $ python runtests.py -v
    $ python runtests.py -v -s random
    $ python runtests.py -v -t numpy/core/tests/test_iter.py:test_iter_c_order
    $ python runtests.py --ipython
    $ python runtests.py --python somescript.py
    $ python runtests.py --bench
    $ python runtests.py -g -m full

This builds Numpy first, so the first time it may take a few minutes.  If
you specify ``-n``, the tests are run against the version of NumPy (if
any) found on current PYTHONPATH.

Using ``runtests.py`` is the recommended approach to running tests.
There are also a number of alternatives to it, for example in-place
build or installing to a virtualenv. See the FAQ below for details.


Building in-place
-----------------

For development, you can set up an in-place build so that changes made to
``.py`` files have effect without rebuild. First, run::

    $ python setup.py build_ext -i

This allows you to import the in-place built NumPy *from the repo base
directory only*.  If you want the in-place build to be visible outside that
base dir, you need to point your ``PYTHONPATH`` environment variable to this
directory.  Some IDEs (Spyder for example) have utilities to manage
``PYTHONPATH``.  On Linux and OSX, you can run the command::

    $ export PYTHONPATH=$PWD

and on Windows::

    $ set PYTHONPATH=/path/to/numpy

Now editing a Python source file in NumPy allows you to immediately
test and use your changes (in ``.py`` files), by simply restarting the
interpreter.

Note that another way to do an inplace build visible outside the repo base dir
is with ``python setup.py develop``.  Instead of adjusting ``PYTHONPATH``, this
installs a ``.egg-link`` file into your site-packages as well as adjusts the
``easy-install.pth`` there, so its a more permanent (and magical) operation.


Other build options
-------------------

It's possible to do a parallel build with ``numpy.distutils`` with the ``-j`` option;
see :ref:`parallel-builds` for more details.

In order to install the development version of NumPy in ``site-packages``, use
``python setup.py install --user``.

A similar approach to in-place builds and use of ``PYTHONPATH`` but outside the
source tree is to use::

    $ python setup.py install --prefix /some/owned/folder
    $ export PYTHONPATH=/some/owned/folder/lib/python3.4/site-packages


Using virtualenvs
-----------------

A frequently asked question is "How do I set up a development version of NumPy
in parallel to a released version that I use to do my job/research?".

One simple way to achieve this is to install the released version in
site-packages, by using a binary installer or pip for example, and set
up the development version in a virtualenv.  First install
`virtualenv`_ (optionally use `virtualenvwrapper`_), then create your
virtualenv (named numpy-dev here) with::

    $ virtualenv numpy-dev

Now, whenever you want to switch to the virtual environment, you can use the
command ``source numpy-dev/bin/activate``, and ``deactivate`` to exit from the
virtual environment and back to your previous shell.


Running tests
-------------

Besides using ``runtests.py``, there are various ways to run the tests.  Inside
the interpreter, tests can be run like this::

    >>> np.test()
    >>> np.test('full')   # Also run tests marked as slow
    >>> np.test('full', verbose=2)   # Additionally print test name/file

Or a similar way from the command line::

    $ python -c "import numpy as np; np.test()"

Tests can also be run with ``nosetests numpy``, however then the NumPy-specific
``nose`` plugin is not found which causes tests marked as ``KnownFailure`` to
be reported as errors.

Running individual test files can be useful; it's much faster than running the
whole test suite or that of a whole module (example: ``np.random.test()``).
This can be done with::

    $ python path_to_testfile/test_file.py

That also takes extra arguments, like ``--pdb`` which drops you into the Python
debugger when a test fails or an exception is raised.

Running tests with `tox`_ is also supported.  For example, to build NumPy and
run the test suite with Python 3.4, use::

    $ tox -e py34

For more extensive info on running and writing tests, see
https://github.com/numpy/numpy/blob/master/doc/TESTS.rst.txt .

*Note: do not run the tests from the root directory of your numpy git repo,
that will result in strange test errors.*


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


Debugging
---------

Another frequently asked question is "How do I debug C code inside Numpy?".
The easiest way to do this is to first write a Python script that invokes the C
code whose execution you want to debug. For instance ``mytest.py``::

    from numpy import linspace
    x = np.arange(5)
    np.empty_like(x)

Now, you can run::

    $ gdb --args python runtests.py -g --python mytest.py

And then in the debugger::

    (gdb) break array_empty_like
    (gdb) run

The execution will now stop at the corresponding C function and you can step
through it as usual.  With the Python extensions for gdb installed (often the
default on Linux), a number of useful Python-specific commands are available.
For example to see where in the Python code you are, use ``py-list``.  For more
details, see `DebuggingWithGdb`_.

Instead of plain ``gdb`` you can of course use your favourite
alternative debugger; run it on the python binary with arguments
``runtests.py -g --python mytest.py``.

Building NumPy with a Python built with debug support (on Linux distributions
typically packaged as ``python-dbg``) is highly recommended.



.. _DebuggingWithGdb: https://wiki.python.org/moin/DebuggingWithGdb

.. _tox: http://tox.testrun.org

.. _virtualenv: http://www.virtualenv.org/

.. _virtualenvwrapper: http://www.doughellmann.com/projects/virtualenvwrapper/

.. _Waf: https://code.google.com/p/waf/
