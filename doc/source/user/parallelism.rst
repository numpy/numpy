Parallelism in NumPy (Overview)
==============================

This page provides a high-level overview of how parallelism relates to NumPy.
It is intended as an introduction and does not provide performance tuning advice.

What “parallelism” means in NumPy
---------------------------------
NumPy is a Python library, but many NumPy operations are implemented in compiled
code. Some operations may execute in parallel internally depending on the
underlying libraries used, while others run serially at the Python level.

The Global Interpreter Lock (GIL)
---------------------------------
The Python Global Interpreter Lock (GIL) ensures that only one thread executes
Python bytecode at a time. Some NumPy operations may release the GIL during
execution, but this behavior depends on the specific operation and
implementation.

When parallelism may or may not help
------------------------------------
Parallel execution can improve performance for certain workloads, especially
large numerical operations. However, parallelism does not guarantee speedups and
may introduce overhead.

Common pitfalls
---------------
Users may encounter issues such as oversubscription or incorrect assumptions
about automatic parallel execution.

Further reading
---------------
- NumPy documentation on performance considerations
- Python documentation on threading and multiprocessing
