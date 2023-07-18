.. _numpy_docs_mainpage:

###################
NumPy documentation
###################

.. toctree::
   :maxdepth: 1
   :hidden:

   User Guide <user/index>
   API reference <reference/index>
   Development <dev/index>
   release


**Version**: |version|

**Download documentation**:
`Historical versions of documentation <https://numpy.org/doc/>`_
   
**Useful links**:
`Installation <https://numpy.org/install/>`_ |
`Source Repository <https://github.com/numpy/numpy>`_ |
`Issue Tracker <https://github.com/numpy/numpy/issues>`_ |
`Q&A Support <https://numpy.org/gethelp/>`_ |
`Mailing List <https://mail.python.org/mailman/listinfo/numpy-discussion>`_

NumPy is the fundamental package for scientific computing in Python. It is a
Python library that provides a multidimensional array object, various derived
objects (such as masked arrays and matrices), and an assortment of routines for
fast operations on arrays, including mathematical, logical, shape manipulation,
sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra,
basic statistical operations, random simulation and much more.

NumPy fully supports an object-oriented approach, starting, once again, with ndarray. 
For example, ndarray is a class, possessing numerous methods and attributes. Many of its 
methods are mirrored by functions in the outer-most NumPy namespace, allowing the programmer 
to code in whichever paradigm they prefer. This flexibility has allowed the NumPy array dialect 
and NumPy ndarray class to become the de-facto language of multi-dimensional data interchange 
used in Python.

#Why is NumPy Fast?
Vectorization describes the absence of any explicit looping, indexing, etc., 
in the code - these things are taking place, of course, just “behind the scenes” 
in optimized, pre-compiled C code. Vectorized code has many advantages, among which are:

* vectorized code is more concise and easier to read

* fewer lines of code generally means fewer bugs

* the code more closely resembles standard mathematical notation (making it easier, typically, to correctly code mathematical constructs)

* vectorization results in more “Pythonic” code. Without vectorization, our code would be littered with inefficient and difficult to read for loops.

.. grid:: 2

    .. grid-item-card::
        :img-top: ../source/_static/index-images/getting_started.svg

        Getting Started
        ^^^^^^^^^^^^^^^

        New to NumPy? Check out the Absolute Beginner's Guide. It contains an
        introduction to NumPy's main concepts and links to additional tutorials.

        +++

        .. button-ref:: user/absolute_beginners
            :expand:
            :color: secondary
            :click-parent:

            To the absolute beginner's guide

    .. grid-item-card::
        :img-top: ../source/_static/index-images/user_guide.svg

        User Guide
        ^^^^^^^^^^

        The user guide provides in-depth information on the
        key concepts of NumPy with useful background information and explanation.

        +++

        .. button-ref:: user
            :expand:
            :color: secondary
            :click-parent:

            To the user guide

    .. grid-item-card::
        :img-top: ../source/_static/index-images/api.svg

        API Reference
        ^^^^^^^^^^^^^

        The reference guide contains a detailed description of the functions,
        modules, and objects included in NumPy. The reference describes how the
        methods work and which parameters can be used. It assumes that you have an
        understanding of the key concepts.

        +++

        .. button-ref:: reference
            :expand:
            :color: secondary
            :click-parent:

            To the reference guide

    .. grid-item-card::
        :img-top: ../source/_static/index-images/contributor.svg

        Contributor's Guide
        ^^^^^^^^^^^^^^^^^^^

        Want to add to the codebase? Can help add translation or a flowchart to the
        documentation? The contributing guidelines will guide you through the
        process of improving NumPy.

        +++

        .. button-ref:: devindex
            :expand:
            :color: secondary
            :click-parent:

            To the contributor's guide

.. This is not really the index page, that is found in
   _templates/indexcontent.html The toctree content here will be added to the
   top of the template header
