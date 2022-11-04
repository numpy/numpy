# Building with Meson

### Developer build

**Install build tools:** `pip install -r build_requirements.txt`
**Compile and install:** `./dev.py build`

This installs into the `build` directory.

To use the package, add it to your PYTHONPATH:

```
export PYTHONPATH=${PWD}/build/lib64/python3.10/site-packages
pytest --pyargs skimage
```

Or launch testing or a shell via `dev.py`:

```
./dev.py test
./dev.py ipython
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

### Building on Fedora

- Fedora does not distribute `openblas.pc`. Install the following file in `~/lib/pkgconfig/openblas.pc`:

```
prefix=/usr
includedir=${prefix}/include
libdir=${prefix}/lib64

Name: openblas
Description: OpenBLAS is an optimized BLAS library based on GotoBLAS2 1.13 BSD version
Version: 0.3.19
Cflags: -I${includedir}/openblas
Libs: -L${libdir} -lopenblas
```

Then build with:

```
./dev.py build -- -Dpkg_config_path=${HOME}/lib/pkgconfig
```
