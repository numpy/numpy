# Building with Meson

### Developer build

**Install build tools:** `pip install -r requirements/build.txt`

**Generate ninja make files:** `meson build --prefix=$PWD/build`

**Compile:** `ninja -C build`

**Install:** `meson install -C build`

The install step copies the necessary Python files into the build dir to form a complete package.
Do not skip this step, or the package won't work.

To use the package, add it to your PYTHONPATH:

```
export PYTHONPATH=${PWD}/build/lib64/python3.10/site-packages
pytest --pyargs skimage
```

### pip install

The standard installation procedure via pip still works:

```
pip install --no-build-isolation .
```

Note, however, that `pip install -e .` (in-place developer install) does not!
See "Developer build" above.

### sdist and wheel

The Python `build` module calls Meson and ninja as necessary to
produce an sdist and a wheel:

```
python -m build --no-isolation
```

## Notes

### Templated Cython files

The `skimage/morphology/skeletonize_3d.pyx.in` is converted into a pyx
file using Tempita. That pyx file appears in the _build_
directory, and can be compiled from there.

If that file had to import local `*.pyx` files (it does not) then the
build dependencies would need be set to ensure that the relevant pyx
files are copied into the build directory prior to compilation (see
`_cython_tree` in the SciPy Meson build files).
