*************
NumPy license
*************

NumPy's source code and documentation are licensed under the
`BSD 3-Clause License <https://opensource.org/license/bsd-3-clause>`__
(SPDX identifier ``BSD-3-Clause``). This is a permissive license: you may use,
modify, and redistribute NumPy, in source or binary form and for any purpose
including commercial use, as long as you retain the copyright notice, the
list of conditions, and the disclaimer below.

The full text of the license is reproduced at the bottom of this page, and is
also available as the ``LICENSE.txt`` file in the
`NumPy repository <https://github.com/numpy/numpy/blob/main/LICENSE.txt>`__.

Vendored components
===================

The NumPy source tree includes a small number of third-party components that
are maintained by other projects and have their own licenses. All of these
are permissive licenses compatible with BSD-3-Clause. They include MIT, 0BSD,
CC0-1.0, and the dual Apache-2.0 / BSD-3-Clause license of Google Highway.
The license file for each vendored component is kept alongside its source in
the repository, and the complete list is recorded under ``license-files`` in
`pyproject.toml <https://github.com/numpy/numpy/blob/main/pyproject.toml>`__.
These files are installed with NumPy so that the terms travel with the code.

Binary wheels on PyPI
=====================

The pre-built wheels that NumPy publishes on
`PyPI <https://pypi.org/project/numpy/>`__ are convenient to install because
they bundle the compiled libraries NumPy depends on at runtime. Those bundled
libraries are **not** covered by NumPy's BSD-3-Clause license; they come with
their own terms, which are included in the wheel:

- ``libopenblas`` (OpenBLAS): ``BSD-3-Clause AND BSD-3-Clause-Attribution``
- ``libgfortran`` (GNU Fortran runtime): ``GPL-3.0-with-GCC-exception``
- ``libquadmath`` (GCC quad-precision math): ``LGPL-2.1-or-later``

Exactly which of these libraries are bundled depends on the platform. See the
license comments in
`pyproject.toml <https://github.com/numpy/numpy/blob/main/pyproject.toml>`__
for the current list.

In practice, the GCC runtime exception and the LGPL permit linking these
libraries into a BSD-licensed program, which is why NumPy can ship them. If
your use case requires that every binary you redistribute is BSD-licensed, you
can avoid the bundled libraries entirely by building NumPy from source against
your own BLAS/LAPACK installation (see :ref:`building-from-source`), or by
installing NumPy from a distribution channel such as conda-forge, where these
libraries are provided as separate packages.

License text
============

.. include:: ../../LICENSE.txt
   :literal:
