..  -*- rst -*-

================
NumPy benchmarks
================

Benchmarking NumPy with Airspeed Velocity.


Usage
-----

Airspeed Velocity manages building and Python virtualenvs by itself,
unless told otherwise. Some of the benchmarking features in
``runtests.py`` also tell ASV to use the NumPy compiled by
``runtests.py``. To run the benchmarks, you do not need to install a
development version of NumPy to your current Python environment.

Before beginning, ensure that *airspeed velocity* is installed.
By default, `asv` ships with support for anaconda and virtualenv::

    pip install asv
    pip install virtualenv

After contributing new benchmarks, you should test them locally
before submitting a pull request.

To run all benchmarks, navigate to the root NumPy directory at
the command line and execute::

    python runtests.py --bench

where ``--bench`` activates the benchmark suite instead of the
test suite. This builds NumPy and runs  all available benchmarks
defined in ``benchmarks/``. (Note: this could take a while. Each
benchmark is run multiple times to measure the distribution in
execution times.)

To run benchmarks from a particular benchmark module, such as
``bench_core.py``, simply append the filename without the extension::

    python runtests.py --bench bench_core

To run a benchmark defined in a class, such as ``Mandelbrot``
from ``bench_avx.py``::

    python runtests.py --bench bench_avx.Mandelbrot

Compare change in benchmark results to another version/commit/branch::

    python runtests.py --bench-compare v1.6.2 bench_core
    python runtests.py --bench-compare 8bf4e9b bench_core
    python runtests.py --bench-compare main bench_core

All of the commands above display the results in plain text in
the console, and the results are not saved for comparison with
future commits. For greater control, a graphical view, and to
have results saved for future comparison you can run ASV commands
(record results and generate HTML)::

    cd benchmarks
    asv run -n -e --python=same
    asv publish
    asv preview

More on how to use ``asv`` can be found in `ASV documentation`_
Command-line help is available as usual via ``asv --help`` and
``asv run --help``.

.. _ASV documentation: https://asv.readthedocs.io/


Writing benchmarks
------------------

See `ASV documentation`_ for basics on how to write benchmarks.

Some things to consider:

- The benchmark suite should be importable with any NumPy version.

- The benchmark parameters etc. should not depend on which NumPy version
  is installed.

- Try to keep the runtime of the benchmark reasonable.

- Prefer ASV's ``time_`` methods for benchmarking times rather than cooking up
  time measurements via ``time.clock``, even if it requires some juggling when
  writing the benchmark.

- Preparing arrays etc. should generally be put in the ``setup`` method rather
  than the ``time_`` methods, to avoid counting preparation time together with
  the time of the benchmarked operation.

- Be mindful that large arrays created with ``np.empty`` or ``np.zeros`` might
  not be allocated in physical memory until the memory is accessed. If this is
  desired behaviour, make sure to comment it in your setup function. If
  you are benchmarking an algorithm, it is unlikely that a user will be
  executing said algorithm on a newly created empty/zero array. One can force
  pagefaults to occur in the setup phase either by calling ``np.ones`` or
  ``arr.fill(value)`` after creating the array,
