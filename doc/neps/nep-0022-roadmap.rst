=============
Numpy Roadmap
=============

:Author: Stéfan van der Walt
:Author: Charles Harris
:Author: Stephan Hoyer
:Author: Matti Picus
:Author: Jarrod Millman
:Author: Nathaniel J Smith
:Author: Matthew Rocklin 

:Status: Draft
:Type: Process Track
:Created: 2018-06-27

Abstract
--------

Detailed description
--------------------

This document was created at the NumPy sprint held at BIDS, 24 May 2018 by the authors.

There was a `discussion <http://numpy-discussion.10968.n7.nabble.com/A-roadmap-for-NumPy-longer-term-planning-td45613.html>`_ on the mailing list. We will revisit this at the upcoming `SciPy conference BOF session <https://scipy2018.scipy.org>`_ 

Work on NumPy can be divided into various tasks, all worthy of devloper time investment. Broadly speaking, we can identify the following tasks

Ongoing Maintenance
~~~~~~~~~~~~~~~~~~~

- triaging issues and pull requests
- document both python and C-API (i.e. borrowed or new refs)

  - can we document reference borrowing close to C code (in the spirit of doctests for Python code)
- code coverage in tests
- actually deprecate things that are deprecated
- make internals private

Policies for Ensuring Code Quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Is this a separate NEP?

- policy for merging pull requests

  - bugfixes - does it fix the bug
  - releases have their own rules
  - at least one reviewer
  - documentation typos can be merged by the committer
  - enhancements should go through the mailing list

New Functionality
~~~~~~~~~~~~~~~~~

- Extensible dtypes
- Overloads for duck arrays [2]_
- Many new NEPs in progress
- `Typing stubs for mypy <https://github.com/numpy/numpy-stubs>`_

  - Basic support in NumPy
  - Standard way to write/check annotations for N-dimensional arrays in Python (this will need a PEP)
- **Internal refactorings** of almost-external things like MaskedArray, f2py, numpy.distutils, financial
- How to handle the **Matrix class**; we want to avoid new usage, reduce external dependency
- Better **configuration reporting** (use it when reporting bugs)
- Allow switching numpy.linalg backends

Lessons Learned
---------------

We may not be able to change any of these Right Now.

- Currently strategy is: either pass to ufunc or convert to ndarray. Makes it hard to use NumPy as high-level API, because it is unknown (without reading the code) whether other classes will survive various operations. See discussion above.
- 0-dim arrays get converted into NumPy scalars

Infrastructure / Ecosystem
--------------------------

- `multibuild <https://github.com/matthew-brett/multibuild>`_ wheel builder (Linux/OSX/Windows)
- Airspeed Speed Velocity for benchmarks (https://pv.github.io/numpy-bench/) or 
  codespeed (https://speed.python.org)
- Maintain the websites scipy.org, numpy.org
- numpydoc
- pytest

  - To what extent should we use pytest APIs?
  - pytest is a little magical (e.g., fixtures)
  - Should define a philosophy. Discussion led to a direction of simple use of pytest, using as little magic as feasable.
- OpenBLAS

Philosophy / scope
------------------

What is part of NumPy, what is not?

- In memory, N-dimensional single pointer + strided arrays on CPUs

  - Not specialized hardware
- Basic functionality for scientific computing

  - Could potentially be split out into separate packages with coordinated releases

    - masked arrays
    - polynomials
    - financial?
    - random numbers

  - Overlapping functionality with SciPy

    - linear algebra
    - FFT
    - Windowing functions for ffts.
    - where is the cutoff? Simple things that everyone needs should be in NumPy, specialized and complicated functions in SciPy. How do we decide which is which?
  - Infrastructrure for data science packages (distutils, f2py, testing)
- Higher level APIs for N-dimensional arrays

  - NumPy is a *de facto* standard for array APIs in Python
  - Protocols like `__array_ufunc__`, `__array_func__`.

Longer term plans
-----------------

These ideas may be used to solicit further funding.

- Missing values
- Labeled arrays

  - In or out?
  - To some extent solved by third-party libraries like xarray and pandas
  - Typing is also a potential solution
- Speed (probably out of scope, given current resources)

  - How important is this? To what extent should we compete with libraries like TensorFlow, etc.
  - Use intrinsics (AVX, SSE, ...)?
  - Establish a known benchmark suite which would serve to quantify the discussion. Any change would be measured against its effect across the entire suite.
  - JIT options

    - Rewriting internals in higher level language?
    - See also Travis's current efforts: `libxnd <https://github.com/plures/xnd>`_ and `Plures <https://github.com/plures>`_ more generally
    - Intermediate LLVM-like language for expressing array computation, can be shared across packages?

Social / Community
------------------

- Goal: grow number of core maintainers

  - Example (above) of documenting C code more carefully to lower barrier to entry
  - Challenge: retention; if we spend time training new contributors, how do we get them to stay 5 years instead of 1 or 2 (grad school, e.g.)
  - Engage with universities or industry (who could e.g. sponsor developer time)

    - Can we have a standard mechanisms whereby industry can engage

      - Sponsorship of: developer time / specific feature / ... ?
      - Could take on self-contained tasks, such as type stubs
      - Companies may benefit from smaller, better scoped packages; they may not
        want to take on "NumPy", but could be willing to engage with a smaller project
        that won't become a time sink. Hypothetical example: company like UK Met
        Office -> package like Masked Arrays
- Goal: more diverse and inclusive contributor community

  - Office hours for interested participants?
  - Sprints for beginners?
- Consideration: if we cannot move forward with features, external packages will
  work around us somehow. I.e., other packages in the ecosystem progress at
  faster rates than NumPy, if NumPy is unable to respond in a timely fashion to
  external needs, then ad-hoc infrastructure develops outside of the project,
  missing out on the benefits of NumPy's central place within the community.

  - It is also quite hard to get balanced input on this specific issue: members
    of community who want to move forward more quickly may not be around any more.
  - Examples:

    - Pandas, which used to rely on / couple more closely with NumPy.
    - https://github.com/dgasmith/opt_einsum
    
Groups we may want to connect with
----------------------------------

- Intel / MKL
- Tensorflow
- CuPy / Chainer
- Consumers: Autograd, Tangent, TensorLy, Optimized Einsum


Discussion
----------

Copyright
---------

This document is placed under the CC0 1.0 Universell (CC0 1.0) Public Domain Dedication [1]_.

References and Footnotes
------------------------

`numpy-grant-planning <https://github.com/njsmith/numpy-grant-planning>`_

.. [1] To the extent possible under law, the person who associated CC0 
   with this work has waived all copyright and related or neighboring
   rights to this work. The CC0 license may be found at
   https://creativecommons.org/publicdomain/zero/1.0/

.. [2] `Notes from a NEP sprint <https://docs.google.com/document/d/10mmyZ2-9GDm4W_5xJIMnbSzxFrD55lJkNsH8F7UB_Fs>`_

Copyright
---------

This document has been placed in the public domain.
