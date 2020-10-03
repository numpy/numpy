.. _how-to-io:

##############################################################################
Reading and writing files
##############################################################################

This page tackles common applications; for the full collection of I/O
routines, see :ref:`routines.io`.


******************************************************************************
Reading text and CSV_ files
******************************************************************************

.. _CSV: https://en.wikipedia.org/wiki/Comma-separated_values

With no missing values
==============================================================================

Use :func:`numpy.loadtxt`.

With missing values
==============================================================================

Use :func:`numpy.genfromtxt`.

:func:`numpy.genfromtxt` will either

  - return a :ref:`masked array<maskedarray.generic>`
    **masking out missing values** (if ``usemask=True``), or

  - **fill in the missing value** with the value specified in
    ``filling_values`` (default is ``np.nan`` for float, -1 for int).

With non-whitespace delimiters
------------------------------------------------------------------------------
::

    >>> print(open("csv.txt").read())  # doctest: +SKIP
    1, 2, 3
    4,, 6
    7, 8, 9


Masked-array output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

    >>> np.genfromtxt("csv.txt", delimiter=",", usemask=True)  # doctest: +SKIP
    masked_array(
      data=[[1.0, 2.0, 3.0],
            [4.0, --, 6.0],
            [7.0, 8.0, 9.0]],
      mask=[[False, False, False],
            [False,  True, False],
            [False, False, False]],
      fill_value=1e+20)

Array output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

    >>> np.genfromtxt("csv.txt", delimiter=",")  # doctest: +SKIP
    array([[ 1.,  2.,  3.],
           [ 4., nan,  6.],
           [ 7.,  8.,  9.]])

Array output, specified fill-in value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

    >>> np.genfromtxt("csv.txt", delimiter=",", dtype=np.int8, filling_values=99)  # doctest: +SKIP
    array([[ 1,  2,  3],
           [ 4, 99,  6],
           [ 7,  8,  9]], dtype=int8)

Whitespace-delimited
-------------------------------------------------------------------------------

:func:`numpy.genfromtxt` can also parse whitespace-delimited data files
that have missing values if

* **Each field has a fixed width**: Use the width as the `delimiter` argument.
  ::

    # File with width=4. The data does not have to be justified (for example,
    # the 2 in row 1), the last column can be less than width (for example, the 6
    # in row 2), and no delimiting character is required (for instance 8888 and 9
    # in row 3)

    >>> f = open("fixedwidth.txt").read()  # doctest: +SKIP
    >>> print(f)  # doctest: +SKIP
    1   2      3
    44      6
    7   88889

    # Showing spaces as ^
    >>> print(f.replace(" ","^"))  # doctest: +SKIP
    1^^^2^^^^^^3
    44^^^^^^6
    7^^^88889

    >>> np.genfromtxt("fixedwidth.txt", delimiter=4)  # doctest: +SKIP
    array([[1.000e+00, 2.000e+00, 3.000e+00],
           [4.400e+01,       nan, 6.000e+00],
           [7.000e+00, 8.888e+03, 9.000e+00]])

* **A special value (e.g. "x") indicates a missing field**: Use it as the
  `missing_values` argument.
  ::

    >>> print(open("nan.txt").read())  # doctest: +SKIP
    1 2 3
    44 x 6
    7  8888 9

    >>> np.genfromtxt("nan.txt", missing_values="x")  # doctest: +SKIP
    array([[1.000e+00, 2.000e+00, 3.000e+00],
           [4.400e+01,       nan, 6.000e+00],
           [7.000e+00, 8.888e+03, 9.000e+00]])

* **You want to skip the rows with missing values**: Set
  `invalid_raise=False`.
  ::

    >>> print(open("skip.txt").read())  # doctest: +SKIP
    1 2   3
    44    6
    7 888 9

    >>> np.genfromtxt("skip.txt", invalid_raise=False)  # doctest: +SKIP
    __main__:1: ConversionWarning: Some errors were detected !
        Line #2 (got 2 columns instead of 3)
    array([[  1.,   2.,   3.],
           [  7., 888.,   9.]])


* **The delimiter whitespace character is different from the whitespace that
  indicates missing data**. For instance, if columns are delimited by ``\t``,
  then missing data will be recognized if it consists of one
  or more spaces.
  ::

    >>> f = open("tabs.txt").read()  # doctest: +SKIP
    >>> print(f)  # doctest: +SKIP
    1       2       3
    44              6
    7       888     9

    # Tabs vs. spaces
    >>> print(f.replace("\t","^"))  # doctest: +SKIP
    1^2^3
    44^ ^6
    7^888^9

    >>> np.genfromtxt("tabs.txt", delimiter="\t", missing_values=" +")  # doctest: +SKIP
    array([[  1.,   2.,   3.],
           [ 44.,  nan,   6.],
           [  7., 888.,   9.]])

******************************************************************************
Read a file in .npy or .npz format
******************************************************************************

Choices:

  - Use :func:`numpy.load`. It can read files generated by any of
    :func:`numpy.save`, :func:`numpy.savez`, or :func:`numpy.savez_compressed`.

  - Use memory mapping. See `numpy.lib.format.open_memmap`.

******************************************************************************
Write to a file to be read back by NumPy
******************************************************************************

Binary
===============================================================================

Use
:func:`numpy.save`, or to store multiple arrays :func:`numpy.savez`
or :func:`numpy.savez_compressed`.

For :ref:`security and portability <how-to-io-pickle-file>`, set
``allow_pickle=False`` unless the dtype contains Python objects, which
requires pickling.

Masked arrays :any:`can't currently be saved <MaskedArray.tofile>`,
nor can other arbitrary array subclasses.

Human-readable
==============================================================================

:func:`numpy.save` and :func:`numpy.savez` create binary files. To **write a
human-readable file**, use :func:`numpy.savetxt`. The array can only be 1- or
2-dimensional, and there's no ` savetxtz` for multiple files.

Large arrays
==============================================================================

See :ref:`how-to-io-large-arrays`.

******************************************************************************
Read an arbitrarily formatted binary file ("binary blob")
******************************************************************************

Use a :doc:`structured array <basics.rec>`.

**Example:**

The ``.wav`` file header is a 44-byte block preceding ``data_size`` bytes of the
actual sound data::

    chunk_id         "RIFF"
    chunk_size       4-byte unsigned little-endian integer
    format           "WAVE"
    fmt_id           "fmt "
    fmt_size         4-byte unsigned little-endian integer
    audio_fmt        2-byte unsigned little-endian integer
    num_channels     2-byte unsigned little-endian integer
    sample_rate      4-byte unsigned little-endian integer
    byte_rate        4-byte unsigned little-endian integer
    block_align      2-byte unsigned little-endian integer
    bits_per_sample  2-byte unsigned little-endian integer
    data_id          "data"
    data_size        4-byte unsigned little-endian integer

The ``.wav`` file header as a NumPy structured dtype::

    wav_header_dtype = np.dtype([
        ("chunk_id", (bytes, 4)), # flexible-sized scalar type, item size 4
        ("chunk_size", "<u4"),    # little-endian unsigned 32-bit integer
        ("format", "S4"),         # 4-byte string, alternate spelling of (bytes, 4)
        ("fmt_id", "S4"),
        ("fmt_size", "<u4"),
        ("audio_fmt", "<u2"),     #
        ("num_channels", "<u2"),  # .. more of the same ...
        ("sample_rate", "<u4"),   #
        ("byte_rate", "<u4"),
        ("block_align", "<u2"),
        ("bits_per_sample", "<u2"),
        ("data_id", "S4"),
        ("data_size", "<u4"),
        #
        # the sound data itself cannot be represented here:
        # it does not have a fixed size
    ])

    header = np.fromfile(f, dtype=wave_header_dtype, count=1)[0]

This ``.wav`` example is for illustration; to read a ``.wav`` file in real
life, use Python's built-in module :mod:`wave`.

(Adapted from Pauli Virtanen, :ref:`advanced_numpy`, licensed
under `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_.)

.. _how-to-io-large-arrays:

******************************************************************************
Write or read large arrays
******************************************************************************

**Arrays too large to fit in memory** can be treated like ordinary in-memory
arrays using memory mapping.

- Raw array data written with :func:`numpy.ndarray.tofile` or
  :func:`numpy.ndarray.tobytes` can be read with :func:`numpy.memmap`::

      array = numpy.memmap("mydata/myarray.arr", mode="r", dtype=np.int16, shape=(1024, 1024))

- Files output by :func:`numpy.save` (that is, using the numpy format) can be read
  using :func:`numpy.load` with the ``mmap_mode`` keyword argument::

      large_array[some_slice] = np.load("path/to/small_array", mmap_mode="r")

Memory mapping lacks features like data chunking and compression; more
full-featured formats and libraries usable with NumPy include:

* **HDF5**: `h5py <https://www.h5py.org/>`_ or `PyTables <https://www.pytables.org/>`_.
* **Zarr**: `here <https://zarr.readthedocs.io/en/stable/tutorial.html#reading-and-writing-data>`_.
* **NetCDF**: :class:`scipy.io.netcdf_file`.

For tradeoffs among memmap, Zarr, and HDF5, see
`pythonspeed.com <https://pythonspeed.com/articles/mmap-vs-zarr-hdf5/>`_.

******************************************************************************
Write files for reading by other (non-NumPy) tools
******************************************************************************

Formats for **exchanging data** with other tools include HDF5, Zarr, and
NetCDF (see :ref:`how-to-io-large-arrays`).

******************************************************************************
Write or read a JSON file
******************************************************************************

NumPy arrays are **not** directly
`JSON serializable <https://github.com/numpy/numpy/issues/12481>`_.


.. _how-to-io-pickle-file:

******************************************************************************
Save/restore using a pickle file
******************************************************************************

Avoid when possible; :doc:`pickles <python:library/pickle>` are not secure
against erroneous or maliciously constructed data.

Use :func:`numpy.save` and :func:`numpy.load`.  Set ``allow_pickle=False``,
unless the array dtype includes Python objects, in which case pickling is
required.

******************************************************************************
Convert from a pandas DataFrame to a NumPy array
******************************************************************************

See :meth:`pandas.DataFrame.to_numpy`.

******************************************************************************
 Save/restore using `~numpy.ndarray.tofile` and `~numpy.fromfile`
******************************************************************************

In general, prefer :func:`numpy.save` and :func:`numpy.load`.

:func:`numpy.ndarray.tofile` and :func:`numpy.fromfile` lose information on
endianness and precision and so are unsuitable for anything but scratch
storage.

