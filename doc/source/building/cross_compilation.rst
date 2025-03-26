Cross compilation
=================

Cross compilation is a complex topic, we only add some hopefully helpful hints
here (for now). As of May 2023, cross-compilation based on ``crossenv`` is
known to work, as used (for example) in conda-forge. Cross-compilation without
``crossenv`` requires some manual overrides. You instruct these overrides by
passing options to ``meson setup`` via `meson-python`_.

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

For more details and the current status around cross compilation, see:

- The state of cross compilation in Python:
  `pypackaging-native key issue page <https://pypackaging-native.github.io/key-issues/cross_compilation/>`__
- Tracking issue for SciPy cross-compilation needs and issues:
  `scipy#14812 <https://github.com/scipy/scipy/issues/14812>`__
