``numpy.emath`` shape-typing and dtype inference
------------------------------------------------
The ``numpy.emath`` functions now infer the shape-type of the returned array from
their inputs. For nested sequences this is limited to 1-d and 2-d inputs, and for
``ndarray`` all shape-types are preserved.

.. code-block:: python

    reveal_type(np.emath.sqrt([-1, 4]))
    # before: ndarray[tuple[Any, ...], dtype[Any]]
    # after:  ndarray[tuple[int], dtype[complex128 | float64]]

    z = np.array([[1.0, -1.0]], dtype=np.complex64)
    reveal_type(np.emath.log(z))
    # before: ndarray[tuple[Any, ...], dtype[complexfloating]]
    # after:  ndarray[tuple[int, int], dtype[complex64]]

Scalar return types are also more precise. For floating-point inputs, the return
type is a union of the corresponding real and complex scalar types, since the
result depends on the input *values*.

.. code-block:: python

    reveal_type(np.emath.sqrt(-1.0))
    # before: Any
    # after:  complex128 | float64
