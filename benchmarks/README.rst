..  -*- rst -*-

================
NumPy benchmarks
================

Benchmarking Numpy with Airspeed Velocity.


Usage
-----

Airspeed Velocity manages building and Python virtualenvs by itself,
unless told otherwise. Some of the benchmarking features in
``runtests.py`` also tell ASV to use the Numpy compiled by
``runtests.py``. To run the benchmarks, you do not need to install a
development version of Numpy to your current Python environment.

Run a benchmark against currently checked out Numpy version (don't
record the result)::

    python runtests.py --bench bench_core

Compare change in benchmark results to another version::

    python runtests.py --bench-compare v1.6.2 bench_core

Run ASV commands (record results and generate HTML)::

    cd benchmarks
    asv run --skip-existing-commits --steps 10 ALL
    asv publish
    asv preview

More on how to use ``asv`` can be found in `ASV documentation`_
Command-line help is available as usual via ``asv --help`` and
``asv run --help``.

.. _ASV documentation: https://spacetelescope.github.io/asv/


Writing benchmarks
------------------

See `ASV documentation`_ for basics on how to write benchmarks.

Some things to consider:

- The benchmark suite should be importable with any Numpy version.

- The benchmark parameters etc. should not depend on which Numpy version
  is installed.

- Try to keep the runtime of the benchmark reasonable.

- Prefer ASV's ``time_`` methods for benchmarking times rather than cooking up
  time measurements via ``time.clock``, even if it requires some juggling when
  writing the benchmark.

- Preparing arrays etc. should generally be put in the ``setup`` method rather
  than the ``time_`` methods, to avoid counting preparation time together with
  the time of the benchmarked operation.
