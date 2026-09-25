Cross compilation
=================

Cross compilation is a complex topic, we only add some hopefully helpful hints
here (for now). As of May 2025, cross-compilation with a Meson cross file as
well as cross-compilation based on ``crossenv`` are known to work. Conda-forge
uses the latter method. Cross-compilation without ``crossenv`` requires passing
build options to ``meson setup`` via `meson-python`_.

.. _meson-python: https://meson-python.readthedocs.io/en/latest/how-to-guides/meson-args.html

All distributions that are known to successfully cross compile NumPy are using
``python -m build`` (``pypa/build``), but using ``pip`` for that should be
possible as well. Here are links to the NumPy "build recipes" on those
distros:

- `Void Linux <https://github.com/void-linux/void-packages/blob/master/srcpkgs/python3-numpy/template>`_
- `Nix <https://github.com/NixOS/nixpkgs/tree/master/pkgs/development/python-modules/numpy>`_
- `Conda-forge <https://github.com/conda-forge/numpy-feedstock/blob/main/recipe/build.sh>`_

See also `Meson's documentation on cross compilation
<https://mesonbuild.com/Cross-compilation.html>`__ to learn what options you
may need to pass to Meson to successfully cross compile.

One possible hiccup is that the build requires running a compiled executable in
order to determine the ``long double`` format for the host platform. This may be
an obstacle, since it requires ``crossenv`` or QEMU to run the host (cross)
Python. To avoid this problem, specify the paths to the relevant directories in
your *cross file*:

.. code:: ini

    [properties]
    longdouble_format = 'IEEE_DOUBLE_LE'

For an example of a cross file needed to cross-compile NumPy, see
`numpy#288861 <https://github.com/numpy/numpy/issues/28861#issuecomment-2844257091>`__.
Putting that together, invoking a cross build with such a cross file, looks like:

.. code:: bash

   $ python -m build --wheel -Csetup-args="--cross-file=aarch64-myos-cross-file.txt"

For more details and the current status around cross compilation, see:

- The state of cross compilation in Python:
  `pypackaging-native key issue page <https://pypackaging-native.github.io/key-issues/cross_compilation/>`__
- The `set of NumPy issues with the "Cross compilation" label <https://github.com/numpy/numpy/issues?q=state%3Aclosed%20label%3A%2238%20-%20Cross%20compilation%22>`__
- Tracking issue for SciPy cross-compilation needs and issues:
  `scipy#14812 <https://github.com/scipy/scipy/issues/14812>`__
