"""
Provides a deep benchmark for ufunc(universal functions).

Usage:
    deep.{name}_{input types}
Example:
    --bench-compare master "deep.square_(h|H)"
"""
import numpy as np
from textwrap import dedent
from functools import lru_cache

@lru_cache(maxsize=1024)
def rand(size, dtype, prevent_overlap=0):
    if dtype == '?':
        return np.random.randint(0, 1, size=size, dtype=dtype)
    elif dtype in 'bBhHiIlLqQ':
        return np.random.randint(1, 127, size=size, dtype=dtype)
    else:
        return np.array(np.random.rand(size), dtype=dtype)

class _Benchmark:
    number = 10
    repeat = (100, 100, 0)

    def setup(self, size, *strides):
        np.seterr(all='ignore')
        self.fn = getattr(np, self.ufunc)
        self.args = [
            rand(size * strides[c], t, c)[::strides[c]]
            for c, t in enumerate(self.arg_types)
        ]

    def time_run(self, *args):
        self.fn(*self.args)

sizes = [256, 4096, 16384]
strides = [1, 2, 5]
klasses = []

for name in dir(np):
    ufunc = getattr(np, name)
    if not isinstance(ufunc, np.ufunc):
        continue
    param_names  = ["size"]
    param_names += ["stride_in%d"  % i for i in range(ufunc.nin)]
    param_names += ["stride_out%d" % i for i in range(ufunc.nout)]
    param_names  = repr(param_names)
    params  = [sizes]
    params += [strides] * (ufunc.nin + ufunc.nout)
    params  = repr(params)

    for tsym in ufunc.types:
        tsym = tsym.split('->')
        if len(tsym) != 2:
            continue
        klasses.append(dedent("""
        class {name}_{dtypes}(_Benchmark):
            params={params}
            param_names={param_names}
            ufunc="{name}"
            arg_types="{arg_types}"
        """.format(
            name=name, dtypes=('_'.join(tsym)).replace('?', 'bool'),
            params=params, param_names=param_names,
            arg_types=tsym[0]+tsym[1]
        )))

exec('\n'.join(klasses))
