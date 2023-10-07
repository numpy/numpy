:orphan:

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

method

.. auto{{ objtype }}:: {{ fullname | replace("numpy.", "numpy::") }}

{# In the fullname (e.g. `numpy.ma.MaskedArray.methodname`), the module name
is ambiguous. Using a `::` separator (e.g. `numpy::ma.MaskedArray.methodname`)
specifies `numpy` as the module name. #}
