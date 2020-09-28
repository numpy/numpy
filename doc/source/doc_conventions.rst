.. _documentation_conventions:

##############################################################################
Documentation conventions
##############################################################################

- Names that look like :func:`numpy.array` are links to detailed
  documentation.

- Examples often include the Python prompt ``>>>``. This is not part of the
  code and will cause an error if typed or pasted into the Python
  shell. It can be safely typed or pasted into the IPython shell; the ``>>>``
  is ignored.

- Examples often use ``np`` as an alias for ``numpy``; that is, they assume
  you've run::

      >>> import numpy as np

- If you're a code contributor writing a docstring, see :ref:`docstring_intro`.

- If you're a writer contributing ordinary (non-docstring) documentation, see
  :ref:`userdoc_guide`.
