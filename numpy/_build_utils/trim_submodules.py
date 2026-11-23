#!/usr/bin/env python3
"""
Remove the files of git submodules that NumPy does not need from an sdist.

Run by ``meson dist``. Every submodule needs an entry in `UNUSED`, so that
none is added without a decision on what the sdist carries of it.
"""
import os
import re
import shutil

# What to remove from each submodule: documentation, tests, benchmarks and
# CI files. An empty list keeps all of it.
UNUSED = {
    'doc/source/_static/scipy-mathjax': [],
    'numpy/_core/src/common/pythoncapi-compat': ['docs', 'tests'],
    'numpy/_core/src/common/unordered_dense': [
        'data', 'doc', 'example', 'notes', 'scripts', 'test',
    ],
    'numpy/_core/src/highway': [
        'docs', 'g3doc', 'hwy/examples', 'hwy/tests',
    ],
    'numpy/_core/src/npysort/x86-simd-sort': [
        'benchmarks', 'examples', 'misc', 'tests',
    ],
    'numpy/_core/src/umath/svml': [],
    'numpy/fft/pocketfft': [],
    'vendored-meson/meson': [
        'docs', 'graphics', 'manual tests', 'test cases', 'unittests',
    ],
}


def main():
    root = os.environ['MESON_DIST_ROOT']
    with open(os.path.join(root, '.gitmodules')) as f:
        submodules = set(re.findall(r'^\s*path\s*=\s*(\S+)', f.read(), re.M))
    if submodules != set(UNUSED):
        changed = ', '.join(sorted(submodules ^ set(UNUSED)))
        raise SystemExit(f"{__file__}: `UNUSED` needs an update for {changed}")

    for path, entries in UNUSED.items():
        for name in entries:
            entry = os.path.join(root, path, name)
            if os.path.isdir(entry) and not os.path.islink(entry):
                shutil.rmtree(entry)
            else:
                # fails if the entry is gone, so that the lists do not go stale
                os.remove(entry)


if __name__ == "__main__":
    main()
