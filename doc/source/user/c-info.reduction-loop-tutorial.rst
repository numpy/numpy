.. _c-api.reduction-loop-tutorial:

***********************************
Adding a reduction loop to a ufunc
***********************************

.. index::
   pair: ufunc; reduction

This tutorial assumes you are already familiar with writing a basic ufunc,
as described in :doc:`c-info.ufunc-tutorial`.

Background
==========

:meth:`~numpy.ufunc.reduce` (and the ``.sum()``/``.prod()``/... methods
built on top of it for functions like :func:`numpy.add` and
:func:`numpy.multiply`) historically only worked for ufuncs with exactly
one output. The trick that makes this possible is that the ordinary
two-input/one-output elementwise loop can be used as the reduction loop:
if the output pointer is aliased to the first input pointer (with stride 0)
and left unmoved, the loop keeps accumulating into the same location as it
walks over the array.

That trick does not generalize to ufuncs with more than one output, because
their forward elementwise loop has a two-in/``nout``-out signature that has
no natural way to alias an accumulator the same way. To support ``reduce``
for such ufuncs, an ``ArrayMethod`` can instead register a *dedicated*
reduction loop through the ``NPY_METH_get_reduction_loop`` slot. Unlike the
forward loop, this loop has an ``(nout + 1)``-in/``nout``-out signature: it
takes the current per-output accumulators plus one streamed input element,
and writes the updated accumulators. The reason we require this signature is
that it naturally collapses down to the nominal two-input/one-output case
when ``nout = 1``. See :c:macro:`NPY_METH_get_reduction_loop` in the
:doc:`ArrayMethod API reference </reference/c-api/array>` for the exact
data/stride layout.

``reduce`` (like ``accumulate`` and ``reduceat``) still requires the ufunc's
forward loop to have exactly two inputs as reducing folds an accumulator and
one new element together at each step, which is only well-defined for a
binary combining operation. This is unrelated to the number of outputs and
was already true before multi-output reduction existed: calling ``reduce``
on a ufunc with ``nin != 2`` (regardless of ``nout``) raises a
:exc:`ValueError` ("... only supported for binary functions"). Only the
*output* arity is generalized here.

If a ufunc has more than one output and its resolved ``ArrayMethod`` does
not register a reduction loop, calling ``.reduce()`` on it raises a
:exc:`TypeError`, e.g. ``numpy.divmod.reduce`` fails since ``divmod``
has no reduction loop.

Example: a two-output ``minimummaximum`` ufunc
===============================================

The example below defines a ``minimummaximum`` ufunc for the ``'f8'``
(``double``) dtype only, computing the running minimum and maximum in a
single reduction pass: ``minimummaximum.reduce(a)`` returns the pair
``(a.min(), a.max())``, computed in one loop over ``a`` rather than in two.

Because ``NPY_METH_get_reduction_loop`` is an ``ArrayMethod`` slot, the
ufunc's loop must be registered through the new-style ``PyArrayMethod_Spec``
API (:c:func:`PyUFunc_AddLoopFromSpec`) rather than the legacy
``PyUFunc_FromFuncAndData`` loop table used in the previous tutorial. The
ufunc itself is still created with :c:func:`PyUFunc_FromFuncAndData`, just
without any loops attached yet. The loop is added separately, via its
``ArrayMethod`` spec.

The forward (elementwise) loop is 2-in/2-out::

    (a, b) -> (min(a, b), max(a, b))

and the reduction loop is 3-in/2-out (``nout = 2`` outputs, so
``nout + 1 = 3`` inputs)::

    (acc_min, acc_max, x) -> (min(acc_min, x), max(acc_max, x))

The place in the code corresponding to the actual computations is marked
with ``/* BEGIN main computation */`` and ``/* END main computation */``.

    .. code-block:: c

        #define PY_SSIZE_T_CLEAN
        #define NPY_TARGET_VERSION NPY_2_6_API_VERSION
        #define NPY_NO_DEPRECATED_API NPY_API_VERSION
        #include <Python.h>
        #include "numpy/ndarraytypes.h"
        #include "numpy/ufuncobject.h"
        #include "numpy/dtype_api.h"

        /*
         * minmax.c
         * A minimal ufunc for a single dtype ('f8') that computes the
         * running minimum and maximum in one pass, and registers a
         * reduction loop for it so that `minimummaximum.reduce(a)` computes
         * the actual minmax.
         */

        static int
        double_minimummaximum_loop(PyArrayMethod_Context *NPY_UNUSED(context),
                char *const data[], npy_intp const dimensions[],
                npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
        {
            npy_intp n = dimensions[0];
            char *in1 = data[0], *in2 = data[1];
            char *out1 = data[2], *out2 = data[3];
            npy_intp is1 = strides[0], is2 = strides[1];
            npy_intp os1 = strides[2], os2 = strides[3];

            for (npy_intp i = 0; i < n; i++) {
                /* BEGIN main computation */
                double a = *(double *)in1;
                double b = *(double *)in2;
                *(double *)out1 = a < b ? a : b;
                *(double *)out2 = a < b ? b : a;
                /* END main computation */
                in1 += is1; in2 += is2; out1 += os1; out2 += os2;
            }
            return 0;
        }

        /*
         * The reduce machinery aliases out_i to acc_i (with stride 0), so
         * this both reads the running accumulators and writes the new ones.
         */
        static int
        double_minimummaximum_reduce_loop(PyArrayMethod_Context *NPY_UNUSED(context),
                char *const data[], npy_intp const dimensions[],
                npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
        {
            npy_intp n = dimensions[0];
            char *acc_min = data[0], *acc_max = data[1], *x = data[2];
            char *out_min = data[3], *out_max = data[4];
            npy_intp s_amin = strides[0], s_amax = strides[1], s_x = strides[2];
            npy_intp s_omin = strides[3], s_omax = strides[4];

            for (npy_intp i = 0; i < n; i++) {
                /* BEGIN main computation */
                double cur_min = *(double *)acc_min;
                double cur_max = *(double *)acc_max;
                double val = *(double *)x;
                *(double *)out_min = val < cur_min ? val : cur_min;
                *(double *)out_max = val > cur_max ? val : cur_max;
                /* END main computation */
                acc_min += s_amin; acc_max += s_amax; x += s_x;
                out_min += s_omin; out_max += s_omax;
            }
            return 0;
        }

        /*
         * get_reduction_loop is called by the reduce machinery in place of
         * get_strided_loop.  It hands back the (nout+1)->nout loop above
         * instead of the forward elementwise one.
         */
        static int
        minimummaximum_get_reduction_loop(PyArrayMethod_Context *NPY_UNUSED(context),
                int NPY_UNUSED(aligned), int NPY_UNUSED(move_references),
                const npy_intp *NPY_UNUSED(strides),
                PyArrayMethod_StridedLoop **out_loop,
                NpyAuxData **out_transferdata,
                NPY_ARRAYMETHOD_FLAGS *flags)
        {
            *out_loop = &double_minimummaximum_reduce_loop;
            *out_transferdata = NULL;
            *flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
            return 0;
        }

        /*
         * Only one loop is registered ('f8'), so the promoter always
         * requests float64 for every input/output that isn't already
         * pinned by `signature` (e.g. via `dtype=`). This is what lets
         * plain Python floats/ints, or other numeric dtypes, be used
         * directly. Without it, only exact `np.float64` inputs would
         * match the loop registered below. A ufunc that registers loops
         * for several dtypes would instead pick the common dtype of its
         * inputs here (see `PyArray_PromoteDTypeSequence`). This one only
         * ever has a single dtype to offer.
         */
        static int
        minimummaximum_promoter(PyObject *NPY_UNUSED(ufunc),
                PyArray_DTypeMeta *const NPY_UNUSED(op_dtypes[]),
                PyArray_DTypeMeta *const signature[],
                PyArray_DTypeMeta *new_op_dtypes[])
        {
            PyArray_Descr *double_descr = PyArray_DescrFromType(NPY_DOUBLE);
            if (double_descr == NULL) {
                return -1;
            }
            PyArray_DTypeMeta *double_dt = NPY_DTYPE(double_descr);

            for (int i = 0; i < 4; i++) {
                PyArray_DTypeMeta *dt = signature[i] != NULL ? signature[i] : double_dt;
                Py_INCREF(dt);
                new_op_dtypes[i] = dt;
            }
            Py_DECREF(double_descr);
            return 0;
        }

        static int
        register_minimummaximum_promoter(PyObject *minimummaximum)
        {
            /* All-None tuple -> catch-all fallback (a concrete loop always wins). */
            PyObject *none_tuple = PyTuple_Pack(
                    4, Py_None, Py_None, Py_None, Py_None);
            if (none_tuple == NULL) {
                return -1;
            }
            PyObject *promoter = PyCapsule_New(
                    (void *)&minimummaximum_promoter,
                    "numpy._ufunc_promoter", NULL);
            if (promoter == NULL) {
                Py_DECREF(none_tuple);
                return -1;
            }
            int res = PyUFunc_AddPromoter(minimummaximum, none_tuple, promoter);
            Py_DECREF(none_tuple);
            Py_DECREF(promoter);
            return res;
        }

        static PyMethodDef MinMaxMethods[] = {
            {NULL, NULL, 0, NULL}
        };

        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "minmax", NULL, -1, MinMaxMethods,
            NULL, NULL, NULL, NULL
        };

        PyMODINIT_FUNC PyInit_minmax(void)
        {
            import_array();
            import_umath();

            PyObject *m = PyModule_Create(&moduledef);
            if (m == NULL) {
                return NULL;
            }

            /* A 2-in/2-out ufunc. Its loops are registered separately below. */
            PyObject *minimummaximum = PyUFunc_FromFuncAndData(
                    NULL, NULL, NULL, 0, 2, 2, PyUFunc_None,
                    "minimummaximum", "minimummaximum_docstring", 0);
            if (minimummaximum == NULL) {
                Py_DECREF(m);
                return NULL;
            }

            PyArray_Descr *double_descr = PyArray_DescrFromType(NPY_DOUBLE);
            PyArray_DTypeMeta *dt = NPY_DTYPE(double_descr);
            PyArray_DTypeMeta *dtypes[4] = {dt, dt, dt, dt};

            PyType_Slot slots[] = {
                {NPY_METH_strided_loop, (void *)&double_minimummaximum_loop},
                {NPY_METH_get_reduction_loop,
                 (void *)&minimummaximum_get_reduction_loop},
                {0, NULL},
            };

            PyArrayMethod_Spec spec = {
                .name = "double_minimummaximum",
                .nin = 2,
                .nout = 2,
                .casting = NPY_NO_CASTING,
                /* Needed for axis=None / multi-axis reductions. */
                .flags = NPY_METH_IS_REORDERABLE,
                .dtypes = dtypes,
                .slots = slots,
            };

            int res = PyUFunc_AddLoopFromSpec(minimummaximum, &spec);
            Py_DECREF(double_descr);
            if (res < 0 || register_minimummaximum_promoter(minimummaximum) < 0
                    || PyModule_AddObject(m, "minimummaximum", minimummaximum) < 0) {
                Py_XDECREF(minimummaximum);
                Py_DECREF(m);
                return NULL;
            }
            return m;
        }

As with the ufuncs in the previous tutorial, the module needs to declare a
dependency on NumPy to build:

.. tab-set::

   .. tab-item:: meson

      Sample ``pyproject.toml`` and ``meson.build``.

      .. code-block:: toml

         [project]
         name = "minmax"
         dependencies = ["numpy"]
         version = "0.1"

         [build-system]
         requires = ["meson-python", "numpy"]
         build-backend = "mesonpy"

      .. code-block:: meson

         project('minmax', 'c')

         py = import('python').find_installation()
         np_dep = dependency('numpy')

         sources = files('minmax.c')

         extension_module = py.extension_module(
           'minmax',
           sources,
           dependencies: [np_dep],
           install: true,
         )

   .. tab-item:: setuptools

      Sample ``pyproject.toml`` and ``setup.py``.

      .. code-block:: toml

         [project]
         name = "minmax"
         dependencies = ["numpy"]
         version = "0.1"

         [build-system]
         requires = ["setuptools", "numpy"]
         build-backend = "setuptools.build_meta"

      .. code-block:: python

         from setuptools import setup, Extension
         from numpy import get_include

         minmax = Extension('minmax',
                            sources=['minmax.c'],
                            include_dirs=[get_include()])

         setup(name='minmax', version='1.0', ext_modules=[minmax])

After the above has been installed, it can be imported and used as
follows. Note that plain Python floats and ints work directly, thanks to
the promoter, no explicit ``np.float64(...)`` wrapping is needed::

    >>> import numpy as np
    >>> import minmax
    >>> minmax.minimummaximum(3.0, 5.0)
    (np.float64(3.0), np.float64(5.0))
    >>> minmax.minimummaximum(3, 5)
    (np.float64(3.0), np.float64(5.0))
    >>> a = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    >>> minmax.minimummaximum.reduce(a)
    (np.float64(1.0), np.float64(9.0))
    >>> b = np.array(a).reshape(2, 4)
    >>> minmax.minimummaximum.reduce(b, axis=None)
    (np.float64(1.0), np.float64(9.0))
    >>> minmax.minimummaximum.reduce(b, axis=1)
    (array([1., 2.]), array([4., 9.]))

Note that ``minimummaximum.reduce`` returns a tuple with one array per
output (``lo, hi = minimummaximum.reduce(a)``), rather than a single array.
If ``initial`` is passed to ``reduce`` on a multi-output ufunc, it can
either be a single scalar (broadcast to seed every output) or a tuple with
one value per output.

Limitations
===========

This example is intentionally minimal and leaves out several things a
real-world implementation, like the built-in :func:`numpy.add` or a fuller
``minimummaximum``, would normally have:

- The promoter always resolves to ``'f8'`` regardless of the input dtypes,
  so e.g. integer input arrays are silently cast to ``float64`` rather than
  keeping their own dtype. A multi-dtype ufunc would instead register a
  loop per dtype and pick the common one, as in the built-in
  :func:`numpy.minimum`/:func:`numpy.maximum`.
- It registers no reduction identity, so ``minimummaximum.reduce`` of an
  empty array raises a :exc:`ValueError`, the same as ``numpy.maximum.reduce``.
  A multi-output ufunc can provide per-output identities with the
  :c:macro:`NPY_METH_get_multi_reduction_initials` slot, the multi-output
  version of :c:macro:`NPY_METH_get_reduction_initial`. Its function fills one
  initial value per output, for example ``+inf`` for the running minimum and
  ``-inf`` for the running maximum. Reducing an empty array, or reducing with
  a ``where=`` mask, then returns those identities instead of raising::

      static int
      minimummaximum_get_multi_reduction_initials(
              PyArrayMethod_Context *NPY_UNUSED(context),
              npy_bool NPY_UNUSED(reduction_is_empty), void **initials)
      {
          *(double *)initials[0] = NPY_INFINITY;   /* min identity */
          *(double *)initials[1] = -NPY_INFINITY;  /* max identity */
          return 1;
      }

  registered alongside the reduction loop as
  ``{NPY_METH_get_multi_reduction_initials, (void *)&minimummaximum_get_multi_reduction_initials}``.
- :meth:`~numpy.ufunc.accumulate`, :meth:`~numpy.ufunc.reduceat`, and
  :meth:`~numpy.ufunc.at` are not supported for multi-output ufuncs yet,
  only :meth:`~numpy.ufunc.reduce` is.

.. seealso::

   * :doc:`ArrayMethod API reference </reference/c-api/array>` for the full
     :c:macro:`NPY_METH_get_reduction_loop` reference documentation.
   * :ref:`ufuncs-basics` for the Python-level behavior of
     :meth:`~numpy.ufunc.reduce` on multi-output ufuncs.
