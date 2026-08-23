.. _how-to-print:

=======================
 Printing NumPy Arrays
=======================


This page explains how to control the formatting of printed NumPy arrays.
Note that these printing options apply only to arrays, not to scalars.

Defining printing options
=========================

Applying settings globally
--------------------------

Use :func:`numpy.set_printoptions` to change printing options for the entire runtime session. To inspect current print settings, use :func:`numpy.get_printoptions`:

    >>> np.set_printoptions(precision=2)
    >>> np.get_printoptions()
    {'edgeitems': 3, 'threshold': 1000, 'floatmode': 'maxprec', 'precision': 2, 'suppress': False, 'linewidth': 75, 'nanstr': 'nan', 'infstr': 'inf', 'sign': '-', 'formatter': None, 'legacy': False, 'override_repr': None}

To restore the default settings, use:

    >>> np.set_printoptions(edgeitems=3, infstr='inf',
    ... linewidth=75, nanstr='nan', precision=8,
    ... suppress=False, threshold=1000, formatter=None)


Applying settings temporarily
-----------------------------

Use :func:`numpy.printoptions` as a context manager to temporarily override print settings within a specific scope:


    >>> arr = np.array([0.155, 0.184, 0.173])
    >>> with np.printoptions(precision=2):
    ...     print(arr)
    [0.15 0.18 0.17]


All keywords that apply to :func:`numpy.set_printoptions` also apply to :func:`numpy.printoptions`.


Changing the number of digits of precision
==========================================

The default number of fractional digits displayed is 8. You can change this number using ``precision`` keyword.

    >>> arr = np.array([0.1, 0.184, 0.17322])
    >>> with np.printoptions(precision=2):
    ...     print(arr)
    [0.1 0.18 0.17]


The ``floatmode`` option determines how the ``precision`` setting is interpreted. 
By default, ``floatmode=maxprec_equal`` displays values with the minimal number of digits needed to uniquely represent them, 
using the same number of digits across all elements.
If you want to show exactly the same number of digits specified by ``precision``, use ``floatmode=fixed``:

    >>> arr = np.array([0.1, 0.184, 0.173], dtype=np.float32)
    >>> with np.printoptions(precision=2, floatmode="fixed"):
    ...     print(arr)
    [0.10 0.18 0.17]


Changing how `nan` and `inf` are displayed
==========================================

By default, `numpy.nan` is displayed as `nan` and `numpy.inf` is displayed as `inf`.
You can override these representations using the ``nanstr`` and ``infstr`` options:

    >>> arr = np.array([np.inf, np.nan, 0])
    >>> with np.printoptions(nanstr="NAN", infstr="INF"):
    ...     print(arr)
    [INF NAN  0.]


Controlling scientific notations
================================

By default, NumPy uses scientific notation when:

- The absolute value of the smallest number is less than ``1e-4``, or
- The ratio of the largest to the smallest absolute value is greater than ``1e3``

    >>> arr = np.array([0.00002, 210000.0, 3.14])
    >>> print(arr)
    [2.00e-05 2.10e+05 3.14e+00]

To suppress scientific notation and always use fixed-point notation, set ``suppress=True``:

    >>> arr = np.array([0.00002, 210000.0, 3.14])
    >>> with np.printoptions(suppress=True):
    ...     print(arr)
    [     0.00002 210000.           3.14   ]



Applying custom formatting functions
====================================

You can apply custom formatting functions to specific or all data types using ``formatter`` keyword.
See :func:`numpy.set_printoptions` for more details on supported format keys.

For example, to format `datetime64` values with a custom function:

    >>> arr = np.array([np.datetime64("2025-01-01"), np.datetime64("2024-01-01")])
    >>> with np.printoptions(formatter={"datetime":lambda x: f"(Year: {x.item().year}, Month: {x.item().month})"}):
    ...     print(arr)
    [(Year: 2025, Month: 1) (Year: 2024, Month: 1)]

