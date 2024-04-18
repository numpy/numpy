import concurrent.futures

import numpy as np


def test_parallel_errstate_creation():
    # if the coercion cache is enabled and not thread-safe, creating
    # RandomState instances simultaneously leads to a data race
    def func(seed):
        np.random.RandomState(seed)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as tpe:
        futures = [tpe.submit(func, i) for i in range(500)]
        for f in futures:
            f.result()
