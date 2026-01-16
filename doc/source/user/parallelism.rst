Parallelism in NumPy (Overview)
===============================

This page provides a high-level overview of how parallelism relates to NumPy.
It is intended as an introduction and does not provide performance tuning advice.

What “parallelism” means in NumPy
---------------------------------
NumPy is a Python library, but many NumPy operations are implemented in compiled
code. Some operations may execute in parallel internally depending on the
underlying libraries used, while others run serially at the Python level.

Example
~~~~~~~

Many NumPy operations appear as a single Python statement but may execute
using multiple threads internally::

    import numpy as np

    a = np.random.rand(10_000_000)
    b = np.random.rand(10_000_000)

    c = a + b

Although this looks like a simple operation in Python, the underlying
implementation may use parallel execution depending on how NumPy was built.

The Global Interpreter Lock (GIL)
---------------------------------
The Python Global Interpreter Lock (GIL) ensures that only one thread executes
Python bytecode at a time. Some NumPy operations may release the GIL during
execution, but this behavior depends on the specific operation and
implementation.

Example
~~~~~~~

Some NumPy operations release the Global Interpreter Lock (GIL), allowing
multiple threads to run concurrently::

    import threading
    import numpy as np

    def work():
        np.dot(
            np.random.rand(1000, 1000),
            np.random.rand(1000, 1000)
        )

    threads = [threading.Thread(target=work) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

Whether this executes in parallel depends on the NumPy build and the linked
linear algebra libraries.

When parallelism may or may not help
------------------------------------
Parallel execution can improve performance for certain workloads, especially
large numerical operations. However, parallelism does not guarantee speedups and
may introduce overhead.

Example
~~~~~~~

Parallelism is more likely to be beneficial for large array operations::

    import numpy as np
    np.sum(np.random.rand(100_000_000))

For very small arrays, the overhead of parallel execution may outweigh
any potential performance benefit.

Common pitfalls
---------------
Users may encounter issues such as oversubscription or incorrect assumptions
about automatic parallel execution.

Example
~~~~~~~

Running multiple parallel NumPy operations at the same time can lead to
oversubscription, where too many threads compete for CPU resources::

    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    def work():
        np.linalg.svd(np.random.rand(2000, 2000))

    with ThreadPoolExecutor(max_workers=4) as ex:
        ex.map(lambda _: work(), range(4))

This may reduce performance instead of improving it, depending on the system
configuration.

Further reading
---------------
- NumPy documentation on performance considerations
- Python documentation on threading and multiprocessing
