==============
Scope of NumPy
==============

Here, we describe aspects of N-d array computation that are within scope for NumPy development. This is *not* an aspirational definition of where NumPy should aim, but instead captures the status quo—areas which we have decided to continue supporting, at least for the time being.

- **In-memory, N-dimensional, homogeneously typed (single pointer + strided) arrays on CPUs**

  - Support for a wide range of data types
  - Not specialized hardware such as GPUs
  - But, do support wide range of CPUs (e.g. ARM, PowerX)

- **Higher level APIs for N-dimensional arrays**

  - NumPy is a *de facto* standard for array APIs in Python
  - Indexing and fast iteration over elements (ufunc)
  - Interoperability protocols with other data container implementations (like `__array_ufunc__`).

- **Python API and a C API** to the ndarray's methods and attributes.

- Other **specialized types or uses of N-dimensional arrays**:

  - Masked arrays
  - Structured arrays (informally known as record arrays)
  - Memory mapped arrays

- Historically, NumPy has included the following **basic functionality
  in support of scientific computation**. We intend to keep supporting
  (but not to expand) what is currently included:

  - Linear algebra
  - Fast Fourier transforms and windowing
  - Pseudo-random number generators
  - Polynomial fitting

- NumPy provides some **infrastructure for other packages in the scientific Python ecosystem**:

  - numpy.distutils (build support for C++, Fortran, BLAS/LAPACK, and other relevant libraries for scientific computing
  - f2py (generating bindings for Fortran code)
  - testing utilities

- **Speed**: we take performance concerns seriously and aim to execute
  operations on large arrays with similar performance as native C
  code. That said, where conflict arises, maintenance and portability take
  precedence over performance. We aim to prevent regressions where
  possible (e.g., through asv).
