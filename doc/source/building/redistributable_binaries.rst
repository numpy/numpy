Building redistributable binaries
=================================

When ``python -m build`` or ``pip wheel`` is used to build a NumPy wheel,
that wheel will rely on external shared libraries (at least for BLAS/LAPACK and
a Fortran compiler runtime library, perhaps other libraries). Such wheels
therefore will only run on the system on which they are built. See
`the pypackaging-native content under "Building and installing or uploading
artifacts" <https://pypackaging-native.github.io/meta-topics/build_steps_conceptual/#building-and-installing-or-uploading-artifacts>`__ for more context on that.

A wheel like that is therefore an intermediate stage to producing a binary that
can be distributed. That final binary may be a wheel - in that case, run
``auditwheel`` (Linux), ``delocate`` (macOS) or ``delvewheel`` (Windows) to
vendor the required shared libraries into the wheel.

The final binary may also be in another packaging format (e.g., a ``.rpm``,
``.deb`` or ``.conda`` package). In that case, there are packaging
ecosystem-specific tools to first install the wheel into a staging area, then
making the extension modules in that install location relocatable (e.g., by
rewriting RPATHs), and then repackaging it into the final package format.

