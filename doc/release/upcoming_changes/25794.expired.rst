* Assigning to the data attribute is disallowed and will raise

* ``np.binary_repr(a, width)`` will raise if width is too small

* Using ``NPY_CHAR`` in ``PyArray_DescrFromType()`` will raise, use
  ``NPY_STRING`` ``NPY_UNICODE``, or ``NPY_VSTRING`` instead.
