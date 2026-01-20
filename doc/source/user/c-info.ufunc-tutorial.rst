**********************
Writing your own ufunc
**********************

| I have the Power!
| --- *He-Man*


.. _`sec:Creating-a-new`:

Creating a new universal function
=================================

.. index::
   pair: ufunc; adding new

Before reading this, it may help to familiarize yourself with the basics
of C extensions for Python by reading/skimming the tutorials in Section 1
of `Extending and Embedding the Python Interpreter
<https://docs.python.org/extending/index.html>`_ and in :doc:`How to extend
NumPy <c-info.how-to-extend>`

The umath module is a computer-generated C-module that creates many
ufuncs. It provides a great many examples of how to create a universal
function. Creating your own ufunc that will make use of the ufunc
machinery is not difficult either. Suppose you have a function that
you want to operate element-by-element over its inputs. By creating a
new ufunc you will obtain a function that handles

- broadcasting

- N-dimensional looping

- automatic type-conversions with minimal memory usage

- optional output arrays

It is not difficult to create your own ufunc. All that is required is
a 1-d loop for each data-type you want to support. Each 1-d loop must
have a specific signature, and only ufuncs for fixed-size data-types
can be used. The function call used to create a new ufunc to work on
built-in data-types is given below. A different mechanism is used to
register ufuncs for user-defined data-types.

In the next several sections we give example code that can be
easily modified to create your own ufuncs. The examples are
successively more complete or complicated versions of the logit
function, a common function in statistical modeling. Logit is also
interesting because, due to the magic of IEEE standards (specifically
IEEE 754), all of the logit functions created below
automatically have the following behavior.

>>> logit(0)
-inf
>>> logit(1)
inf
>>> logit(2)
nan
>>> logit(-2)
nan

This is wonderful because the function writer doesn't have to
manually propagate infs or nans.

.. _`sec:Non-numpy-example`:

Example non-ufunc extension
===========================

.. index::
   pair: ufunc; adding new

For comparison and general edification of the reader we provide
a simple implementation of a C extension of ``logit`` that uses no
numpy.

To do this we need three files. The first is the C file which contains
the actual code, and the others are two project files that describe
how to create the module.

    .. code-block:: c

        #define PY_SSIZE_T_CLEAN
        #include <Python.h>
        #include <math.h>

        /*
         * spammodule.c
         * This is the C code for a non-numpy Python extension to
         * define the logit function, where logit(p) = log(p/(1-p)).
         * This function will not work on numpy arrays automatically.
         * numpy.vectorize must be called in python to generate
         * a numpy-friendly function.
         *
         * Details explaining the Python-C API can be found under
         * 'Extending and Embedding' and 'Python/C API' at
         * docs.python.org .
         */


        /* This declares the logit function */
        static PyObject *spam_logit(PyObject *self, PyObject *args);

        /*
         * This tells Python what methods this module has.
         * See the Python-C API for more information.
         */
        static PyMethodDef SpamMethods[] = {
            {"logit",
                spam_logit,
                METH_VARARGS, "compute logit"},
            {NULL, NULL, 0, NULL}
        };

        /*
         * This actually defines the logit function for
         * input args from Python.
         */

        static PyObject *spam_logit(PyObject *self, PyObject *args)
        {
            double p;

            /* This parses the Python argument into a double */
            if(!PyArg_ParseTuple(args, "d", &p)) {
                return NULL;
            }

            /* THE ACTUAL LOGIT FUNCTION */
            p = p/(1-p);
            p = log(p);

            /*This builds the answer back into a python object */
            return Py_BuildValue("d", p);
        }

        /* This initiates the module using the above definitions. */
        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "spam",
            NULL,
            -1,
            SpamMethods,
            NULL,
            NULL,
            NULL,
            NULL
        };

        PyMODINIT_FUNC PyInit_spam(void)
        {
            PyObject *m;
            m = PyModule_Create(&moduledef);
            if (!m) {
                return NULL;
            }
            return m;
        }

To create the module, one proceeds as one would for a Python package, creating
a ``pyproject.toml`` file, which defines a build back-end, and then another
file for that backend which describes how to compile the code. For the backend,
we recommend ``meson-python``, as we use it for numpy itself, but below we
also show how to use the older ``setuptools``.

.. tab-set::

   .. tab-item:: meson

      Sample ``pyproject.toml`` and ``meson.build``.

      .. code-block:: toml

         [project]
         name = "spam"
         version = "0.1"

         [build-system]
         requires = ["meson-python"]
         build-backend = "mesonpy"

      .. code-block:: meson

         project('spam', 'c')

         py = import('python').find_installation()

         sources = files('spammodule.c')

         extension_module = py.extension_module(
           'spam',
           sources,
           install: true,
         )

   .. tab-item:: setuptools

      Sample ``pyproject.toml`` and ``setup.py``.

      .. code-block:: toml

         [project]
         name = "spam"
         version = "0.1"

         [build-system]
         requires = ["setuptools"]
         build-backend = "setuptools.build_meta"

      .. code-block:: python

         from setuptools import setup, Extension

         spammodule = Extension('spam', sources=['spammodule.c'])

         setup(name='spam', version='1.0',
               ext_modules=[spammodule])

With either of the above, one can build and install the ``spam`` package with,

.. code-block:: bash

   pip install .

Once the ``spam`` module is imported into python, you can call logit
via ``spam.logit``. Note that the function used above cannot be applied
as-is to numpy arrays. To do so we must call :py:func:`numpy.vectorize`
on it. For example::

    >>> import numpy as np
    >>> import spam
    >>> spam.logit(0)
    -inf
    >>> spam.logit(1)
    inf
    >>> spam.logit(0.5)
    0.0
    >>> x = np.linspace(0,1,10)
    >>> spam.logit(x)
    TypeError: only length-1 arrays can be converted to Python scalars
    >>> f = np.vectorize(spam.logit)
    >>> f(x)
    array([       -inf, -2.07944154, -1.25276297, -0.69314718, -0.22314355,
        0.22314355,  0.69314718,  1.25276297,  2.07944154,         inf])

THE RESULTING LOGIT FUNCTION IS NOT FAST! ``numpy.vectorize`` simply
loops over ``spam.logit``. The loop is done at the C level, but the numpy
array is constantly being parsed and build back up. This is expensive.
When the author compared ``numpy.vectorize(spam.logit)`` against the
logit ufuncs constructed below, the logit ufuncs were almost exactly
4 times faster. Larger or smaller speedups are, of course, possible
depending on the nature of the function.


.. _`sec:NumPy-one-loop`:

Example NumPy ufunc for one dtype
=================================

.. index::
   pair: ufunc; adding new

For simplicity we give a ufunc for a single dtype, the ``'f8'``
``double``. As in the previous section, we first give the ``.c`` file
and then the files used to create a ``npufunc`` module containing the ufunc.

The place in the code corresponding to the actual computations for
the ufunc are marked with ``/* BEGIN main ufunc computation */`` and
``/* END main ufunc computation */``. The code in between those lines is
the primary thing that must be changed to create your own ufunc.

    .. code-block:: c

        #define PY_SSIZE_T_CLEAN
        #include <Python.h>
        #include "numpy/ndarraytypes.h"
        #include "numpy/ufuncobject.h"
        #include "numpy/npy_3kcompat.h"
        #include <math.h>

        /*
         * single_type_logit.c
         * This is the C code for creating your own
         * NumPy ufunc for a logit function.
         *
         * In this code we only define the ufunc for
         * a single dtype. The computations that must
         * be replaced to create a ufunc for
         * a different function are marked with BEGIN
         * and END.
         *
         * Details explaining the Python-C API can be found under
         * 'Extending and Embedding' and 'Python/C API' at
         * docs.python.org .
         */

        static PyMethodDef LogitMethods[] = {
            {NULL, NULL, 0, NULL}
        };

        /* The loop definition must precede the PyMODINIT_FUNC. */

        static void double_logit(char **args, const npy_intp *dimensions,
                                 const npy_intp *steps, void *data)
        {
            npy_intp i;
            npy_intp n = dimensions[0];
            char *in = args[0], *out = args[1];
            npy_intp in_step = steps[0], out_step = steps[1];

            double tmp;

            for (i = 0; i < n; i++) {
                /* BEGIN main ufunc computation */
                tmp = *(double *)in;
                tmp /= 1 - tmp;
                *((double *)out) = log(tmp);
                /* END main ufunc computation */

                in += in_step;
                out += out_step;
            }
        }

        /* This a pointer to the above function */
        PyUFuncGenericFunction funcs[1] = {&double_logit};

        /* These are the input and return dtypes of logit.*/
        static const char types[2] = {NPY_DOUBLE, NPY_DOUBLE};

        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "npufunc",
            NULL,
            -1,
            LogitMethods,
            NULL,
            NULL,
            NULL,
            NULL
        };

        PyMODINIT_FUNC PyInit_npufunc(void)
        {
            PyObject *m, *logit, *d;

            import_array();
            import_umath();

            m = PyModule_Create(&moduledef);
            if (!m) {
                return NULL;
            }

            logit = PyUFunc_FromFuncAndData(funcs, NULL, types, 1, 1, 1,
                                            PyUFunc_None, "logit",
                                            "logit_docstring", 0);

            d = PyModule_GetDict(m);

            PyDict_SetItemString(d, "logit", logit);
            Py_DECREF(logit);

            return m;
        }

For the files needed to create the module, the main difference from our
previous example is that we now need to declare dependencies on numpy.

.. tab-set::

   .. tab-item:: meson

      Sample ``pyproject.toml`` and ``meson.build``.

      .. code-block:: toml

         [project]
         name = "npufunc"
         dependencies = ["numpy"]
         version = "0.1"

         [build-system]
         requires = ["meson-python", "numpy"]
         build-backend = "mesonpy"

      .. code-block:: meson

         project('npufunc', 'c')

         py = import('python').find_installation()
         np_dep = dependency('numpy')

         sources = files('single_type_logit.c')

         extension_module = py.extension_module(
           'npufunc',
           sources,
           dependencies: [np_dep],
           install: true,
         )

   .. tab-item:: setuptools

      Sample ``pyproject.toml`` and ``setup.py``.

      .. code-block:: toml

         [project]
         name = "npufunc"
         dependencies = ["numpy"]
         version = "0.1"

         [build-system]
         requires = ["setuptools", "numpy"]
         build-backend = "setuptools.build_meta"

      .. code-block:: python

         from setuptools import setup, Extension
         from numpy import get_include

         npufunc = Extension('npufunc',
                             sources=['single_type_logit.c'],
                             include_dirs=[get_include()])

         setup(name='npufunc', version='1.0', ext_modules=[npufunc])

After the above has been installed, it can be imported and used as follows::

    >>> import numpy as np
    >>> import npufunc
    >>> npufunc.logit(0.5)
    np.float64(0.0)
    >>> a = np.linspace(0, 1, 5)
    >>> npufunc.logit(a)
    array([       -inf, -1.09861229,  0.        ,  1.09861229,         inf])


.. _`sec:NumPy-many-loop`:

Example NumPy ufunc with multiple dtypes
========================================

.. index::
   pair: ufunc; adding new

We now extend the above to a full ``logit`` ufunc, with inner loops for
floats, doubles, and long doubles. Here, we can use the same build files
as above, except we need to change the source file from ``single_type_logit.c``
to ``multi_type_logit.c``.

The places in the code corresponding to the actual computations for
the ufunc are marked with ``/* BEGIN main ufunc computation */`` and
``/* END main ufunc computation */``. The code in between those lines
is the primary thing that must be changed to create your own ufunc.


    .. code-block:: c

        #define PY_SSIZE_T_CLEAN
        #include <Python.h>
        #include "numpy/ndarraytypes.h"
        #include "numpy/ufuncobject.h"
        #include <math.h>

        /*
         * multi_type_logit.c
         * This is the C code for creating your own
         * NumPy ufunc for a logit function.
         *
         * Each function of the form type_logit defines the
         * logit function for a different numpy dtype. Each
         * of these functions must be modified when you
         * create your own ufunc. The computations that must
         * be replaced to create a ufunc for
         * a different function are marked with BEGIN
         * and END.
         *
         * Details explaining the Python-C API can be found under
         * 'Extending and Embedding' and 'Python/C API' at
         * docs.python.org .
         *
         */

        static PyMethodDef LogitMethods[] = {
            {NULL, NULL, 0, NULL}
        };

        /* The loop definitions must precede the PyMODINIT_FUNC. */

        static void long_double_logit(char **args, const npy_intp *dimensions,
                                      const npy_intp *steps, void *data)
        {
            npy_intp i;
            npy_intp n = dimensions[0];
            char *in = args[0], *out = args[1];
            npy_intp in_step = steps[0], out_step = steps[1];

            long double tmp;

            for (i = 0; i < n; i++) {
                /* BEGIN main ufunc computation */
                tmp = *(long double *)in;
                tmp /= 1 - tmp;
                *((long double *)out) = logl(tmp);
                /* END main ufunc computation */

                in += in_step;
                out += out_step;
            }
        }

        static void double_logit(char **args, const npy_intp *dimensions,
                                 const npy_intp *steps, void *data)
        {
            npy_intp i;
            npy_intp n = dimensions[0];
            char *in = args[0], *out = args[1];
            npy_intp in_step = steps[0], out_step = steps[1];

            double tmp;

            for (i = 0; i < n; i++) {
                /* BEGIN main ufunc computation */
                tmp = *(double *)in;
                tmp /= 1 - tmp;
                *((double *)out) = log(tmp);
                /* END main ufunc computation */

                in += in_step;
                out += out_step;
            }
        }

        static void float_logit(char **args, const npy_intp *dimensions,
                               const npy_intp *steps, void *data)
        {
            npy_intp i;
            npy_intp n = dimensions[0];
            char *in = args[0], *out = args[1];
            npy_intp in_step = steps[0], out_step = steps[1];

            float tmp;

            for (i = 0; i < n; i++) {
                /* BEGIN main ufunc computation */
                tmp = *(float *)in;
                tmp /= 1 - tmp;
                *((float *)out) = logf(tmp);
                /* END main ufunc computation */

                in += in_step;
                out += out_step;
            }
        }



        /*This gives pointers to the above functions*/
        PyUFuncGenericFunction funcs[3] = {&float_logit,
                                           &double_logit,
                                           &long_double_logit};

        static const char types[6] = {NPY_FLOAT, NPY_FLOAT,
                                      NPY_DOUBLE, NPY_DOUBLE,
                                      NPY_LONGDOUBLE, NPY_LONGDOUBLE};

        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "npufunc",
            NULL,
            -1,
            LogitMethods,
            NULL,
            NULL,
            NULL,
            NULL
        };

        PyMODINIT_FUNC PyInit_npufunc(void)
        {
            PyObject *m, *logit, *d;

            import_array();
            import_umath();

            m = PyModule_Create(&moduledef);
            if (!m) {
                return NULL;
            }

            logit = PyUFunc_FromFuncAndData(funcs, NULL, types, 4, 1, 1,
                                            PyUFunc_None, "logit",
                                            "logit_docstring", 0);

            d = PyModule_GetDict(m);

            PyDict_SetItemString(d, "logit", logit);
            Py_DECREF(logit);

            return m;
        }

After the above has been installed, it can be imported and used as follows.

>>> import numpy as np
>>> import npufunc
>>> npufunc.logit(0.5)
np.float64(0.0)
>>> a = np.linspace(0, 1, 5, dtype=np.float32)
>>> npufunc.logit(a)
<python-input-4>:1: RuntimeWarning: divide by zero encountered in logit
array([      -inf, -1.0986123,  0.       ,  1.0986123,        inf],
      dtype=float32)

.. note::

   Supporting ``float16`` (half-precision) in custom ufuncs is more complex
   due to its non-standard C representation and conversion requirements.  The
   above code can process ``float16`` input, but will do so by converting it
   to ``float32``. The result will then be ``float32`` too, but one can
   convert it back to ``float16`` by passing in a suitable output, as in
   ``npufunc.logit(a, out=np.empty_like(a))``. For examples of actual
   ``float16`` loops, see the numpy source code.

.. _`sec:NumPy-many-arg`:

Example NumPy ufunc with multiple arguments/return values
=========================================================

Creating a ufunc with multiple arguments is not difficult. Here, we make a
modification of the code for a logit ufunc, where we compute ``(A * B,
logit(A * B))``. For simplicity, we only create a loop for doubles.

We again only give the C code as the files needed to create the module are the
same as before, but with the source file name replaced by
``multi_arg_logit.c``.

The C file is given below. The ufunc generated takes two arguments ``A``
and ``B``. It returns a tuple whose first element is ``A * B`` and whose second
element is ``logit(A * B)``. Note that it automatically supports broadcasting,
as well as all other properties of a ufunc.

    .. code-block:: c

        #define PY_SSIZE_T_CLEAN
        #include <Python.h>
        #include "numpy/ndarraytypes.h"
        #include "numpy/ufuncobject.h"
        #include <math.h>

        /*
         * multi_arg_logit.c
         * This is the C code for creating your own
         * NumPy ufunc for a multiple argument, multiple
         * return value ufunc. The places where the
         * ufunc computation is carried out are marked
         * with comments.
         *
         * Details explaining the Python-C API can be found under
         * 'Extending and Embedding' and 'Python/C API' at
         * docs.python.org.
         */

        static PyMethodDef LogitMethods[] = {
            {NULL, NULL, 0, NULL}
        };

        /* The loop definition must precede the PyMODINIT_FUNC. */

        static void double_logitprod(char **args, const npy_intp *dimensions,
                                     const npy_intp *steps, void *data)
        {
            npy_intp i;
            npy_intp n = dimensions[0];
            char *in1 = args[0], *in2 = args[1];
            char *out1 = args[2], *out2 = args[3];
            npy_intp in1_step = steps[0], in2_step = steps[1];
            npy_intp out1_step = steps[2], out2_step = steps[3];

            double tmp;

            for (i = 0; i < n; i++) {
                /* BEGIN main ufunc computation */
                tmp = *(double *)in1;
                tmp *= *(double *)in2;
                *((double *)out1) = tmp;
                *((double *)out2) = log(tmp / (1 - tmp));
                /* END main ufunc computation */

                in1 += in1_step;
                in2 += in2_step;
                out1 += out1_step;
                out2 += out2_step;
            }
        }

        /*This a pointer to the above function*/
        PyUFuncGenericFunction funcs[1] = {&double_logitprod};

        /* These are the input and return dtypes of logit.*/

        static const char types[4] = {NPY_DOUBLE, NPY_DOUBLE,
                                      NPY_DOUBLE, NPY_DOUBLE};

        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "npufunc",
            NULL,
            -1,
            LogitMethods,
            NULL,
            NULL,
            NULL,
            NULL
        };

        PyMODINIT_FUNC PyInit_npufunc(void)
        {
            PyObject *m, *logit, *d;

            import_array();
            import_umath();

            m = PyModule_Create(&moduledef);
            if (!m) {
                return NULL;
            }

            logit = PyUFunc_FromFuncAndData(funcs, NULL, types, 1, 2, 2,
                                            PyUFunc_None, "logit",
                                            "logit_docstring", 0);

            d = PyModule_GetDict(m);

            PyDict_SetItemString(d, "logit", logit);
            Py_DECREF(logit);

            return m;
        }


.. _`sec:NumPy-struct-dtype`:

Example NumPy ufunc with structured array dtype arguments
=========================================================

This example shows how to create a ufunc for a structured array dtype.
For the example we show a trivial ufunc for adding two arrays with dtype
``'u8,u8,u8'``. The process is a bit different from the other examples since
a call to :c:func:`PyUFunc_FromFuncAndData` cannot register ufuncs for
custom dtypes and structured array dtypes. We need to also call
:c:func:`PyUFunc_RegisterLoopForDescr` to finish setting up the ufunc.

We only give the C code as the files needed to construct the module are again
exactly the same as before, except that the source file is now ``add_triplet.c``.

    .. code-block:: c

        #define PY_SSIZE_T_CLEAN
        #include <Python.h>
        #include "numpy/ndarraytypes.h"
        #include "numpy/ufuncobject.h"
        #include "numpy/npy_3kcompat.h"
        #include <math.h>

        /*
         * add_triplet.c
         * This is the C code for creating your own
         * NumPy ufunc for a structured array dtype.
         *
         * Details explaining the Python-C API can be found under
         * 'Extending and Embedding' and 'Python/C API' at
         * docs.python.org.
         */

        static PyMethodDef StructUfuncTestMethods[] = {
            {NULL, NULL, 0, NULL}
        };

        /* The loop definition must precede the PyMODINIT_FUNC. */

        static void add_uint64_triplet(char **args, const npy_intp *dimensions,
                                       const npy_intp *steps, void *data)
        {
            npy_intp i;
            npy_intp is1 = steps[0];
            npy_intp is2 = steps[1];
            npy_intp os = steps[2];
            npy_intp n = dimensions[0];
            uint64_t *x, *y, *z;

            char *i1 = args[0];
            char *i2 = args[1];
            char *op = args[2];

            for (i = 0; i < n; i++) {

                x = (uint64_t *)i1;
                y = (uint64_t *)i2;
                z = (uint64_t *)op;

                z[0] = x[0] + y[0];
                z[1] = x[1] + y[1];
                z[2] = x[2] + y[2];

                i1 += is1;
                i2 += is2;
                op += os;
            }
        }

        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "npufunc",
            NULL,
            -1,
            StructUfuncTestMethods,
            NULL,
            NULL,
            NULL,
            NULL
        };

        PyMODINIT_FUNC PyInit_npufunc(void)
        {
            PyObject *m, *add_triplet, *d;
            PyObject *dtype_dict;
            PyArray_Descr *dtype;
            PyArray_Descr *dtypes[3];

            import_array();
            import_umath();

            m = PyModule_Create(&moduledef);
            if (m == NULL) {
                return NULL;
            }

            /* Create a new ufunc object */
            add_triplet = PyUFunc_FromFuncAndData(NULL, NULL, NULL, 0, 2, 1,
                                                  PyUFunc_None, "add_triplet",
                                                  "add_triplet_docstring", 0);

            dtype_dict = Py_BuildValue("[(s, s), (s, s), (s, s)]",
                                       "f0", "u8", "f1", "u8", "f2", "u8");
            PyArray_DescrConverter(dtype_dict, &dtype);
            Py_DECREF(dtype_dict);

            dtypes[0] = dtype;
            dtypes[1] = dtype;
            dtypes[2] = dtype;

            /* Register ufunc for structured dtype */
            PyUFunc_RegisterLoopForDescr((PyUFuncObject *)add_triplet,
                                         dtype,
                                         &add_uint64_triplet,
                                         dtypes,
                                         NULL);

            d = PyModule_GetDict(m);

            PyDict_SetItemString(d, "add_triplet", add_triplet);
            Py_DECREF(add_triplet);
            return m;
        }

Sample usage::

    >>> import npufunc
    >>> import numpy as np
    >>> a = np.array([(1, 2, 3), (4, 5, 6)], "u8,u8,u8")
    >>> npufunc.add_triplet(a, a)
    array([(2,  4,  6), (8, 10, 12)],
          dtype=[('f0', '<u8'), ('f1', '<u8'), ('f2', '<u8')])
