.. _advanced_debugging:

========================
Advanced debugging tools
========================

If you reached here, you want to dive into, or use, more advanced tooling.
This is usually not necessary for first time contributors and most
day-to-day development.
These are used more rarely, for example close to a new NumPy release,
or when a large or particular complex change was made.

Since not all of these tools are used on a regular basis and only available
on some systems, please expect differences, issues, or quirks;
we will be happy to help if you get stuck and appreciate any improvements
or suggestions to these workflows.


Finding C errors with additional tooling
########################################

Most development will not require more than a typical debugging toolchain
as shown in :ref:`Debugging <debugging>`. 
But for example memory leaks can be particularly subtle or difficult to
narrow down.

We do not expect any of these tools to be run by most contributors.
However, you can ensure that we can track down such issues more easily:

* Tests should cover all code paths, including error paths.
* Try to write short and simple tests. If you have a very complicated test
  consider creating an additional simpler test as well.
  This can be helpful, because often it is only easy to find which test
  triggers an issue and not which line of the test.
* Never use ``np.empty`` if data is read/used. ``valgrind`` will notice this
  and report an error. When you do not care about values, you can generate
  random values instead.

This will help us catch any oversights before your change is released
and means you do not have to worry about making reference counting errors,
which can be intimidating.


Python debug build
==================

Debug builds of Python are easily available for example via the system package
manager on Linux systems, but are also available on other platforms, possibly in
a less convenient format. If you cannot easily install a debug build of Python
from a system package manager, you can build one yourself using `pyenv
<https://github.com/pyenv/pyenv>`_. For example, to install and globally
activate a debug build of Python 3.13.3, one would do::

    pyenv install -g 3.13.3
    pyenv global 3.13.3

Note that ``pyenv install`` builds Python from source, so you must ensure that
Python's dependencies are installed before building, see the pyenv documentation
for platform-specific installation instructions. You can use ``pip`` to install
Python dependencies you may need for your debugging session. If there is no
debug wheel available on `pypi,` you will need to build the dependencies from
source and ensure that your dependencies are also compiled as debug builds.

Often debug builds of Python name the Python executable ``pythond`` instead of
``python``. To check if you have a debug build of Python installed, you can run
e.g. ``pythond -m sysconfig`` to get the build configuration for the Python
executable. A debug build will be built with debug compiler options in
``CFLAGS`` (e.g. ``-g -Og``).

Running the Numpy tests or an interactive terminal is usually as easy as::

    python3.8d runtests.py
    # or
    python3.8d runtests.py --ipython

and were already mentioned in :ref:`Debugging <debugging>`.

A Python debug build will help:

- Find bugs which may otherwise cause random behaviour.
  One example is when an object is still used after it has been deleted.

- Python debug builds allows to check correct reference counting.
  This works using the additional commands::

    sys.gettotalrefcount()
    sys.getallocatedblocks()

- Python debug builds allow easier debugging with gdb and other C debuggers.


Use together with ``pytest``
----------------------------

Running the test suite only with a debug python build will not find many
errors on its own. An additional advantage of a debug build of Python is that
it allows detecting memory leaks.

A tool to make this easier is `pytest-leaks`_, which can be installed using ``pip``.
Unfortunately, ``pytest`` itself may leak memory, but good results can usually
(currently) be achieved by removing::

    @pytest.fixture(autouse=True)
    def add_np(doctest_namespace):
        doctest_namespace['np'] = numpy

    @pytest.fixture(autouse=True)
    def env_setup(monkeypatch):
        monkeypatch.setenv('PYTHONHASHSEED', '0')

from ``numpy/conftest.py`` (This may change with new ``pytest-leaks`` versions
or ``pytest`` updates).

This allows to run the test suite, or part of it, conveniently::

    python3.8d runtests.py -t numpy/_core/tests/test_multiarray.py -- -R2:3 -s

where ``-R2:3`` is the ``pytest-leaks`` command (see its documentation), the
``-s`` causes output to print and may be necessary (in some versions captured
output was detected as a leak).

Note that some tests are known (or even designed) to leak references, we try
to mark them, but expect some false positives.

.. _pytest-leaks: https://github.com/abalkin/pytest-leaks

``valgrind``
============

Valgrind is a powerful tool to find certain memory access problems and should
be run on complicated C code.
Basic use of ``valgrind`` usually requires no more than::

    PYTHONMALLOC=malloc valgrind python runtests.py

where ``PYTHONMALLOC=malloc`` is necessary to avoid false positives from python
itself.
Depending on the system and valgrind version, you may see more false positives.
``valgrind`` supports "suppressions" to ignore some of these, and Python does
have a suppression file (and even a compile time option) which may help if you
find it necessary.

Valgrind helps:

- Find use of uninitialized variables/memory.

- Detect memory access violations (reading or writing outside of allocated
  memory).

- Find *many* memory leaks. Note that for *most* leaks the python
  debug build approach (and ``pytest-leaks``) is much more sensitive.
  The reason is that ``valgrind`` can only detect if memory is definitely
  lost. If::

      dtype = np.dtype(np.int64)
      arr.astype(dtype=dtype)

  Has incorrect reference counting for ``dtype``, this is a bug, but valgrind
  cannot see it because ``np.dtype(np.int64)`` always returns the same object.
  However, not all dtypes are singletons, so this might leak memory for
  different input.
  In rare cases NumPy uses ``malloc`` and not the Python memory allocators
  which are invisible to the Python debug build.
  ``malloc`` should normally be avoided, but there are some exceptions
  (e.g. the ``PyArray_Dims`` structure is public API and cannot use the
  Python allocators.)

Even though using valgrind for memory leak detection is slow and less sensitive
it can be a convenient: you can run most programs with valgrind without
modification.

Things to be aware of:

- Valgrind does not support the numpy ``longdouble``, this means that tests
  will fail or be flagged errors that are completely fine.

- Expect some errors before and after running your NumPy code.

- Caches can mean that errors (specifically memory leaks) may not be detected
  or are only detect at a later, unrelated time.

A big advantage of valgrind is that it has no requirements aside from valgrind
itself (although you probably want to use debug builds for better tracebacks).


Use together with ``pytest``
----------------------------
You can run the test suite with valgrind which may be sufficient
when you are only interested in a few tests::

    PYTHONMALLOC=malloc valgrind python runtests.py \
     -t numpy/_core/tests/test_multiarray.py -- --continue-on-collection-errors

Note the ``--continue-on-collection-errors``, which is currently necessary due to
missing ``longdouble`` support causing failures (this will usually not be
necessary if you do not run the full test suite).

If you wish to detect memory leaks you will also require ``--show-leak-kinds=definite``
and possibly more valgrind options.  Just as for ``pytest-leaks`` certain
tests are known to leak cause errors in valgrind and may or may not be marked
as such.

We have developed `pytest-valgrind`_ which:

- Reports errors for each test individually

- Narrows down memory leaks to individual tests (by default valgrind
  only checks for memory leaks after a program stops, which is very
  cumbersome).

Please refer to its ``README`` for more information (it includes an example
command for NumPy).

.. _pytest-valgrind: https://github.com/seberg/pytest-valgrind

