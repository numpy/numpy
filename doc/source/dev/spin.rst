.. _spin_tool:

Spin: NumPy’s developer tool
----------------------------

NumPy uses a command-line tool called ``spin`` to support common development
tasks such as building from source, running tests, building documentation, 
and managing other
developer workflows.

The ``spin`` tool provides a consistent interface for contributors working on
NumPy itself, wrapping multiple underlying tools and configurations into a
single command that follows NumPy’s development conventions.
Running the full test suite::

    $ spin test -m full

Running a subset of tests::

    $ spin test -t numpy/_core/tests

Running tests with coverage::

    $ spin test --coverage

Building the documentation::

    $ spin docs