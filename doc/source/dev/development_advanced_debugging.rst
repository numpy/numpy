.. _advanced_debugging:

========================
Advanced debugging tools
========================

If you reached here, you want to dive into, or use, more advanced tooling.
This is usually not necessary for first-time contributors and most
day-to-day development.
These are used more rarely, for example close to a new NumPy release,
or when a large or particular complex change was made.

Some of these tools are used in NumPy's continuous integration tests. If you
see a test failure that only happens under a debugging tool, these instructions
should hopefully enable you to reproduce the test failure locally.

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
* Never use ``np.empty`` if data is read/used.
  `Valgrind <https://valgrind.org/>`_ will notice this
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

`Valgrind <https://valgrind.org/>`_ is a powerful tool
to find certain memory access problems and should
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
it can be convenient: you can run most programs with valgrind without
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


C debuggers
===========

Whenever NumPy crashes or when working on changes to NumPy's low-level C or C++
code, it's often convenient to run Python under a C debugger to get more
information. A debugger can aid in understanding an interpreter crash (e.g. due
to a segmentation fault) by providing a C call stack at the site of the
crash. The call stack often provides valuable context to understand the nature
of a crash. C debuggers are also very useful during development, allowing
interactive debugging in the C implementation of NumPy.

The NumPy developers often use both ``gdb`` and ``lldb`` to debug Numpy. As a
rule of thumb, ``gdb`` is often easier to use on Linux while ``lldb`` is easier
to use on a Mac environment. They have disjoint user interfaces, so you will need to
learn how to use whichever one you land on. The ``gdb`` to ``lldb`` `command map
<https://lldb.llvm.org/use/map.html>`_ is a convenient reference for how to
accomplish common recipes in both debuggers.


Building With Debug Symbols
---------------------------

The ``spin`` `development workflow tool
<https://github.com/scientific-python/spin>`_. has built-in support for working
with both ``gdb`` and ``lldb`` via the ``spin gdb`` and ``spin lldb`` commands.

.. note::

   Building with ``-Dbuildtype=debug`` has a couple of important effects to
   be aware of:

   * **Assertions are enabled**: This build type does not define the ``NDEBUG``
     macro, which means that any C-level assertions in the code will be
     active. This is very useful for debugging, as it can help pinpoint
     where an unexpected condition occurs.

   * **Compiler flags may need overriding**: Some compiler toolchains,
     particularly those from ``conda-forge``, may set optimization flags
     like ``-O2`` by default. These can override the ``debug`` build type.
     To ensure a true debug build in such environments, you may need to
     manually unset or override this flag.

   For more details on both points, see the `meson-python guide on
   debug builds <https://mesonbuild.com/meson-python/how-to-guides/debug-builds.html>`_.

For both debuggers, it's advisable to build NumPy in either the ``debug`` or
``debugoptimized`` meson build profile. To use ``debug`` you can pass the option
via ``spin build``:

.. code-block:: bash

   spin build -- -Dbuildtype=debug

to use ``debugoptimized`` you're pass ``-Dbuildtype=debugoptimized`` instead.

You can pass additional arguments to `meson setup
<https://mesonbuild.com/Builtin-options.html>`_ besides ``buildtype`` using the
same positional argument syntax for ``spin build``.

Running a Test Script
---------------------

Let's say you have a test script named `test.py` that lives in a ``test`` folder
in the same directory as the NumPy source checkout. You could execute the test
script using the ``spin`` build of NumPy with the following incantation:

.. code-block:: bash

   spin gdb ../test/test.py

This will launch into gdb. If all you care about is a call stack for a crash,
type "r" and hit enter. Your test script will run and if a crash happens, you
type "bt" to get a traceback. For ``lldb``, the instructions are similar, just
replace ``spin gdb`` with ``spin lldb``.

You can also set breakpoints and use other more advanced techniques. See the
documentation for your debugger for more details.

One common issue with breakpoints in NumPy is that some code paths get hit
repeatedly during the import of the ``numpy`` module. This can make it tricky or
tedious to find the first "real" call after the NumPy import has completed and
the ``numpy`` module is fully initialized.

One workaround is to use a script like this:

.. code-block:: python

   import os
   import signal

   import numpy as np

   PID = os.getpid()

   def do_nothing(*args):
       pass

   signal.signal(signal.SIGUSR1, do_nothing)

   os.kill(PID, signal.SIGUSR1)

   # the code to run under a debugger follows


This example installs a signal handler for the ``SIGUSR1`` signal that does
nothing and then calls ``os.kill`` on the Python process with the ``SIGUSR1``
signal. This causes the signal handler to fire and critically also causes both
``gdb`` and ``lldb`` to halt execution inside of the ``kill`` syscall.

If you run ``lldb`` you should see output something like this:

.. code-block::

   Process 67365 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGUSR1
        frame #0: 0x000000019c4b9da4 libsystem_kernel.dylib`__kill + 8
    libsystem_kernel.dylib`__kill:
    ->  0x19c4b9da4 <+8>:  b.lo   0x19c4b9dc4    ; <+40>
        0x19c4b9da8 <+12>: pacibsp
        0x19c4b9dac <+16>: stp    x29, x30, [sp, #-0x10]!
        0x19c4b9db0 <+20>: mov    x29, sp
    Target 0: (python3.13) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGUSR1
      * frame #0: 0x000000019c4b9da4 libsystem_kernel.dylib`__kill + 8
        frame #1: 0x000000010087f5c4 libpython3.13.dylib`os_kill + 104
        frame #2: 0x000000010071374c libpython3.13.dylib`cfunction_vectorcall_FASTCALL + 276
        frame #3: 0x00000001006c1e3c libpython3.13.dylib`PyObject_Vectorcall + 88
        frame #4: 0x00000001007edd1c libpython3.13.dylib`_PyEval_EvalFrameDefault + 23608
        frame #5: 0x00000001007e7e6c libpython3.13.dylib`PyEval_EvalCode + 252
        frame #6: 0x0000000100852944 libpython3.13.dylib`run_eval_code_obj + 180
        frame #7: 0x0000000100852610 libpython3.13.dylib`run_mod + 220
        frame #8: 0x000000010084fa4c libpython3.13.dylib`_PyRun_SimpleFileObject + 868
        frame #9: 0x000000010084f400 libpython3.13.dylib`_PyRun_AnyFileObject + 160
        frame #10: 0x0000000100874ab8 libpython3.13.dylib`pymain_run_file + 336
        frame #11: 0x0000000100874324 libpython3.13.dylib`Py_RunMain + 1516
        frame #12: 0x000000010087459c libpython3.13.dylib`pymain_main + 324
        frame #13: 0x000000010087463c libpython3.13.dylib`Py_BytesMain + 40
        frame #14: 0x000000019c152b98 dyld`start + 6076
   (lldb)

As you can see, the C stack trace is inside of the ``kill`` syscall and an
``lldb`` prompt is active, allowing interactively setting breakpoints. Since the
``os.kill`` call happens after the ``numpy`` module is already fully
initialized, this means any breakpoints set inside of ``kill`` will happen
*after* ``numpy`` is finished initializing.

Use together with ``pytest``
----------------------------

You can also run ``pytest`` tests under a debugger. This requires using
the debugger in a slightly more manual fashion, since ``spin`` does not yet
automate this process. First, run ``spin build`` to ensure there is a fully
built copy of NumPy managed by ``spin``. Then, to run the tests under ``lldb``
you would do something like this:

.. code-block:: bash

   spin lldb $(which python) $(which pytest) build-install/usr/lib/python3.13/site-packages/numpy/_core/tests/test_multiarray.py

This will execute the tests in ``test_multiarray.py`` under lldb after typing
'r' and hitting enter. Note that this command comes from a session using Python
3.13 on a Mac. If you are using a different Python version or operating system,
the directory layout inside ``build-install`` may be slightly different.

You can set breakpoints as described above. The issue about breakpoints
commonly being hit during NumPy import also applies - consider refactoring your
test workflow into a test script so you can adopt the workaround using
``os.kill`` described above.

Note the use of ``$(which python)`` to ensure the debugger receives a path to a
Python executable. If you are using ``pyenv``, you may need to replace ``which
python`` with ``pyenv which python``, since ``pyenv`` relies on shim scripts
that ``which`` doesn't know about.


Compiler Sanitizers
===================

The `compiler sanitizer <https://hpc-wiki.info/hpc/Compiler_Sanitizers>`_ suites
shipped by both GCC and LLVM offer a means to detect many common programming
errors at runtime. The sanitizers work by instrumenting the application code at
build time so additional runtime checks fire. Typically, sanitizers are run
during the course of regular testing and if a sanitizer check fails, this leads
to a test failure or crash, along with a report about the nature of the failure.

While it is possible to use sanitizers with a "regular" build of CPython - it is
best if you can set up a Python environment based on a from-source Python build
with sanitizer instrumentation, and then use the instrumented Python to build
NumPy and run the tests. If the entire Python stack is instrumented using the
same sanitizer runtime, it becomes possible to identify issues that happen
across the Python stack. This enables detecting memory leaks in NumPy due to
misuse of memory allocated in CPython, for example.

Build Python with Sanitizer Instrumentation
-------------------------------------------

See the `section in the Python developer's guide
<https://devguide.python.org/getting-started/setup-building/>`_ on this topic for
more information about building Python from source. To enable address sanitizer,
you will need to pass ``--with-address-sanitizer`` to the ``configure`` script
invocation when you build Python.

You can also use `pyenv <https://github.com/pyenv/pyenv>`_ to automate the
process of building Python and quickly activate or deactivate a Python
installation using a command-line interface similar to virtual
environments. With ``pyenv`` you could install an ASAN-instrumented build of
Python 3.13 like this:

.. code-block:: bash

   CONFIGURE_OPTS="--with-address-sanitizer" pyenv install 3.13

If you are interested in thread sanitizer, the ``cpython_sanity`` `docker images
<https://github.com/nascheme/cpython_sanity>`_ might also be a quicker choice
that bypasses building Python from source, although it may be annoying to do
debugging work inside of a docker image.

Use together with ``spin``
--------------------------

However you build Python, once you have an instrumented Python build, you can
install NumPy's development and test dependencies and build NumPy with address
sanitizer instrumentation. For example, to build NumPy with the ``debug``
profile and address sanitizer, you would pass additional build options to
``meson`` like this:

.. code-block:: bash

   spin build -- -Dbuildtype=debug -Db_sanitize=address


Once the build is finished, you can use other ``spin`` command like ``spin
test`` and ``spin gdb`` as with any other Python build.

Special considerations
----------------------

Some NumPy tests intentionally lead to ``malloc`` returning ``NULL``. In its
default configuration, some of the compiler sanitizers flag this as an
error. You can disable that check by passing ``allocator_may_return_null=1`` to
the sanitizer as an option. For example, with address sanitizer:

.. code-block:: bash

   ASAN_OPTIONS=allocator_may_return_null=1 spin test

You may see memory leaks coming from the Python interpreter, particularly on
MacOS. If the memory leak reports are not useful, you can disable leak detection
by passing ``detect_leaks=0`` in ``ASAN_OPTIONS``. You can pass more than one
option using a colon-delimited list, like this:

.. code-block:: bash

   ASAN_OPTIONS=allocator_may_return_null=1:halt_on_error=1:detect_leaks=1 spin test

The ``halt_on_error`` option can be particularly useful -- it hard-crashes the
Python executable whenever it detects an error, along with a report about the
error that includes a stack trace.

You can also take a look at the ``compiler_sanitizers.yml`` GitHub actions
workflow configuration. It describes several different CI jobs that are run as
part of the NumPy tests using Thread, Address, and Undefined Behavior sanitizer.
