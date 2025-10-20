NumPy/SciPy testing guidelines
==============================

.. contents::


Introduction
''''''''''''

Until the 1.15 release, NumPy used the `nose`_ testing framework, it now uses
the `pytest`_ framework. The older framework is still maintained in order to
support downstream projects that use the old numpy framework, but all tests
for NumPy should use pytest.

Our goal is that every module and package in NumPy
should have a thorough set of unit
tests. These tests should exercise the full functionality of a given
routine as well as its robustness to erroneous or unexpected input
arguments. Well-designed tests with good coverage make
an enormous difference to the ease of refactoring. Whenever a new bug
is found in a routine, you should write a new test for that specific
case and add it to the test suite to prevent that bug from creeping
back in unnoticed.

.. note::

  SciPy uses the testing framework from :mod:`numpy.testing`,
  so all of the NumPy examples shown below are also applicable to SciPy

Testing NumPy
'''''''''''''

NumPy can be tested in a number of ways, choose any way you feel comfortable.

Running tests from inside Python
--------------------------------

You can test an installed NumPy by `numpy.test`, for example,
To run NumPy's full test suite, use the following::

  >>> import numpy
  >>> numpy.test(label='slow')

The test method may take two or more arguments; the first ``label`` is a
string specifying what should be tested and the second ``verbose`` is an
integer giving the level of output verbosity. See the docstring
`numpy.test`
for details. The default value for ``label`` is 'fast' - which
will run the standard tests.  The string 'full' will run the full battery
of tests, including those identified as being slow to run. If ``verbose``
is 1 or less, the tests will just show information messages about the tests
that are run; but if it is greater than 1, then the tests will also provide
warnings on missing tests. So if you want to run every test and get
messages about which modules don't have tests::

  >>> numpy.test(label='full', verbose=2)  # or numpy.test('full', 2)

Finally, if you are only interested in testing a subset of NumPy, for
example, the ``_core`` module, use the following::

  >>> numpy._core.test()

Running tests from the command line
-----------------------------------

If you want to build NumPy in order to work on NumPy itself, use the ``spin``
utility. To run NumPy's full test suite::

  $ spin test -m full

Testing a subset of NumPy::

  $ spin test -t numpy/_core/tests

For detailed info on testing, see :ref:`testing-builds`

Running tests in multiple threads
---------------------------------

To help with stress testing NumPy for thread safety, the test suite can be run under
`pytest-run-parallel`_. To install ``pytest-run-parallel``::

  $ pip install pytest-run-parallel

To run the test suite in multiple threads::

  $ spin test -p auto # have pytest-run-parallel detect the number of available cores
  $ spin test -p 4 # run each test under 4 threads
  $ spin test -p auto -- --skip-thread-unsafe=true # run ONLY tests that are thread-safe

When you write new tests (see below), it is worth testing to make sure they do not fail
under ``pytest-run-parallel``, since the CI jobs makes use of it.

Running doctests
----------------

NumPy documentation contains code examples, "doctests". To check that the examples
are correct, install the ``scipy-doctest`` package::

  $ pip install scipy-doctest

and run one of::

  $ spin check-docs -v
  $ spin check-docs numpy/linalg
  $ spin check-docs -- -k 'det and not slogdet'

Note that the doctests are not run when you use ``spin test``.


Other methods of running tests
------------------------------

Run tests using your favourite IDE such as `vscode`_ or `pycharm`_

Writing your own tests
''''''''''''''''''''''

If you are writing code that you'd like to become part of NumPy,
please write the tests as you develop your code.
Every Python module, extension module, or subpackage in the NumPy
package directory should have a corresponding ``test_<name>.py`` file.
Pytest examines these files for test methods (named ``test*``) and test
classes (named ``Test*``).

Suppose you have a NumPy module ``numpy/xxx/yyy.py`` containing a
function ``zzz()``.  To test this function you would create a test
module called ``test_yyy.py``.  If you only need to test one aspect of
``zzz``, you can simply add a test function::

  def test_zzz():
      assert zzz() == 'Hello from zzz'

More often, we need to group a number of tests together, so we create
a test class::

  import pytest

  # import xxx symbols
  from numpy.xxx.yyy import zzz
  import pytest

  class TestZzz:
      def test_simple(self):
          assert zzz() == 'Hello from zzz'

      def test_invalid_parameter(self):
          with pytest.raises(ValueError, match='.*some matching regex.*'):
              ...

Within these test methods, the ``assert`` statement or a specialized assertion
function is used to test whether a certain assumption is valid. If the
assertion fails, the test fails. Common assertion functions include:

- :func:`numpy.testing.assert_equal` for testing exact elementwise equality
  between a result array and a reference,
- :func:`numpy.testing.assert_allclose` for testing near elementwise equality
  between a result array and a reference (i.e. with specified relative and
  absolute tolerances), and
- :func:`numpy.testing.assert_array_less` for testing (strict) elementwise
  ordering between a result array and a reference.

By default, these assertion functions only compare the numerical values in the
arrays. Consider using the ``strict=True`` option to check the array dtype
and shape, too.

When you need custom assertions, use the Python ``assert`` statement. Note that
``pytest`` internally rewrites ``assert`` statements to give informative
output when it fails, so it should be preferred over the legacy variant
``numpy.testing.assert_``. Whereas plain ``assert`` statements are ignored
when running Python in optimized mode with ``-O``, this is not an issue when
running tests with pytest.

Similarly, the pytest functions :func:`pytest.raises` and :func:`pytest.warns`
should be preferred over their legacy counterparts
:func:`numpy.testing.assert_raises` and :func:`numpy.testing.assert_warns`,
which are more broadly used. These versions also accept a ``match``
parameter, which should always be used to precisely target the intended
warning or error.

Note that ``test_`` functions or methods should not have a docstring, because
that makes it hard to identify the test from the output of running the test
suite with ``verbose=2`` (or similar verbosity setting).  Use plain comments
(``#``) to describe the intent of the test and help the unfamiliar reader to
interpret the code.

Also, since much of NumPy is legacy code that was
originally written without unit tests, there are still several modules
that don't have tests yet. Please feel free to choose one of these
modules and develop tests for it.

Using C code in tests
---------------------

NumPy exposes a rich :ref:`C-API<c-api>` . These are tested using c-extension
modules written "as-if" they know nothing about the internals of NumPy, rather
using the official C-API interfaces only. Examples of such modules are tests
for a user-defined ``rational`` dtype in ``_rational_tests`` or the ufunc
machinery tests in ``_umath_tests`` which are part of the binary distribution.
Starting from version 1.21, you can also write snippets of C code in tests that
will be compiled locally into c-extension modules and loaded into python.

.. currentmodule:: numpy.testing.extbuild

.. autofunction:: build_and_import_extension

Labeling tests
--------------

Unlabeled tests like the ones above are run in the default
``numpy.test()`` run.  If you want to label your test as slow - and
therefore reserved for a full ``numpy.test(label='full')`` run, you
can label it with ``pytest.mark.slow``::

  import pytest

  @pytest.mark.slow
  def test_big(self):
      print('Big, slow test')

Similarly for methods::

  class test_zzz:
      @pytest.mark.slow
      def test_simple(self):
          assert_(zzz() == 'Hello from zzz')

Easier setup and teardown functions / methods
---------------------------------------------

Testing looks for module-level or class method-level setup and teardown
functions by name; thus::

  def setup_module():
      """Module-level setup"""
      print('doing setup')

  def teardown_module():
      """Module-level teardown"""
      print('doing teardown')


  class TestMe:
      def setup_method(self):
          """Class-level setup"""
          print('doing setup')

      def teardown_method():
          """Class-level teardown"""
          print('doing teardown')


Setup and teardown functions to functions and methods are known as "fixtures",
and they should be used sparingly.
``pytest`` supports more general fixture at various scopes which may be used
automatically via special arguments.  For example,  the special argument name
``tmpdir`` is used in test to create a temporary directory.

Parametric tests
----------------

One very nice feature of ``pytest`` is the ease of testing across a range
of parameter values using the ``pytest.mark.parametrize`` decorator. For example,
suppose you wish to test ``linalg.solve`` for all combinations of three
array sizes and two data types::

  @pytest.mark.parametrize('dimensionality', [3, 10, 25])
  @pytest.mark.parametrize('dtype', [np.float32, np.float64])
  def test_solve(dimensionality, dtype):
      np.random.seed(842523)
      A = np.random.random(size=(dimensionality, dimensionality)).astype(dtype)
      b = np.random.random(size=dimensionality).astype(dtype)
      x = np.linalg.solve(A, b)
      eps = np.finfo(dtype).eps
      assert_allclose(A @ x, b, rtol=eps*1e2, atol=0)
      assert x.dtype == np.dtype(dtype)

Doctests
--------

Doctests are a convenient way of documenting the behavior of a function
and allowing that behavior to be tested at the same time.  The output
of an interactive Python session can be included in the docstring of a
function, and the test framework can run the example and compare the
actual output to the expected output.

The doctests can be run by adding the ``doctests`` argument to the
``test()`` call; for example, to run all tests (including doctests)
for numpy.lib::

>>> import numpy as np
>>> np.lib.test(doctests=True)

The doctests are run as if they are in a fresh Python instance which
has executed ``import numpy as np``. Tests that are part of a NumPy
subpackage will have that subpackage already imported. E.g. for a test
in ``numpy/linalg/tests/``, the namespace will be created such that
``from numpy import linalg`` has already executed.


``tests/``
----------

Rather than keeping the code and the tests in the same directory, we
put all the tests for a given subpackage in a ``tests/``
subdirectory. For our example, if it doesn't already exist you will
need to create a ``tests/`` directory in ``numpy/xxx/``. So the path
for ``test_yyy.py`` is ``numpy/xxx/tests/test_yyy.py``.

Once the ``numpy/xxx/tests/test_yyy.py`` is written, its possible to
run the tests by going to the ``tests/`` directory and typing::

  python test_yyy.py

Or if you add ``numpy/xxx/tests/`` to the Python path, you could run
the tests interactively in the interpreter like this::

  >>> import test_yyy
  >>> test_yyy.test()

``__init__.py`` and ``setup.py``
--------------------------------

Usually, however, adding the ``tests/`` directory to the python path
isn't desirable. Instead it would better to invoke the test straight
from the module ``xxx``. To this end, simply place the following lines
at the end of your package's ``__init__.py`` file::

  ...
  def test(level=1, verbosity=1):
      from numpy.testing import Tester
      return Tester().test(level, verbosity)

You will also need to add the tests directory in the configuration
section of your setup.py::

  ...
  def configuration(parent_package='', top_path=None):
      ...
      config.add_subpackage('tests')
      return config
  ...

Now you can do the following to test your module::

  >>> import numpy
  >>> numpy.xxx.test()

Also, when invoking the entire NumPy test suite, your tests will be
found and run::

  >>> import numpy
  >>> numpy.test()
  # your tests are included and run automatically!

Tips & Tricks
'''''''''''''

Known failures & skipping tests
-------------------------------

Sometimes you might want to skip a test or mark it as a known failure,
such as when the test suite is being written before the code it's
meant to test, or if a test only fails on a particular architecture.

To skip a test, simply use ``skipif``::

  import pytest

  @pytest.mark.skipif(SkipMyTest, reason="Skipping this test because...")
  def test_something(foo):
      ...

The test is marked as skipped if ``SkipMyTest`` evaluates to nonzero,
and the message in verbose test output is the second argument given to
``skipif``.  Similarly, a test can be marked as a known failure by
using ``xfail``::

  import pytest

  @pytest.mark.xfail(MyTestFails, reason="This test is known to fail because...")
  def test_something_else(foo):
      ...

Of course, a test can be unconditionally skipped or marked as a known
failure by using ``skip`` or ``xfail`` without argument, respectively.

A total of the number of skipped and known failing tests is displayed
at the end of the test run.  Skipped tests are marked as ``'S'`` in
the test results (or ``'SKIPPED'`` for ``verbose > 1``), and known
failing tests are marked as ``'x'`` (or ``'XFAIL'`` if ``verbose >
1``).

Tests on random data
--------------------

Tests on random data are good, but since test failures are meant to expose
new bugs or regressions, a test that passes most of the time but fails
occasionally with no code changes is not helpful. Make the random data
deterministic by setting the random number seed before generating it.  Use
either Python's ``random.seed(some_number)`` or NumPy's
``numpy.random.seed(some_number)``, depending on the source of random numbers.

Alternatively, you can use `Hypothesis`_ to generate arbitrary data.
Hypothesis manages both Python's and Numpy's random seeds for you, and
provides a very concise and powerful way to describe data (including
``hypothesis.extra.numpy``, e.g. for a set of mutually-broadcastable shapes).

The advantages over random generation include tools to replay and share
failures without requiring a fixed seed, reporting *minimal* examples for
each failure, and better-than-naive-random techniques for triggering bugs.


Documentation for ``numpy.test``
--------------------------------

.. autofunction:: numpy.test

.. _nose: https://nose.readthedocs.io/en/latest/
.. _pytest: https://pytest.readthedocs.io
.. _parameterization: https://docs.pytest.org/en/latest/parametrize.html
.. _Hypothesis: https://hypothesis.readthedocs.io/en/latest/
.. _vscode: https://code.visualstudio.com/docs/python/testing#_enable-a-test-framework
.. _pycharm: https://www.jetbrains.com/help/pycharm/testing-your-first-python-application.html
.. _pytest-run-parallel: https://github.com/Quansight-Labs/pytest-run-parallel
