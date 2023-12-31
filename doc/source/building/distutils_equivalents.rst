.. _distutils-meson-equivalents:

Meson and ``distutils`` ways of doing things
--------------------------------------------

*Old workflows (numpy.distutils based):*

1. ``python runtests.py``
2. ``python setup.py build_ext -i`` + ``export
   PYTHONPATH=/home/username/path/to/numpy/reporoot`` (and then edit pure
   Python code in NumPy and run it with ``python some_script.py``).
3. ``python setup.py develop`` - this is similar to (2), except in-place build
   is made permanently visible in env.
4. ``python setup.py bdist_wheel`` + ``pip install dist/numpy*.whl`` - build
   wheel in current env and install it.
5. ``pip install .`` - build wheel in an isolated build env against deps in
   ``pyproject.toml`` and install it. *Note: be careful, this is usually not
   the correct command for development installs - typically you want to use (4)
   or* ``pip install . -v --no-build-isolation``.

*New workflows (Meson and meson-python based):*

1. ``spin test``
2. ``pip install -e . --no-build-isolation`` (note: only for working on NumPy
   itself - for more details, see
   :ref:`IDE support & editable installs <meson-editable-installs>`)
3. the same as (2)
4. ``python -m build --no-isolation`` + ``pip install dist/numpy*.whl`` - see
   `pypa/build <https://pypa-build.readthedocs.io/en/latest/>`_.
5. ``pip install .``
