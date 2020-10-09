{% if objtype == 'property' %}
:orphan:
{% endif %}

{{ fullname | escape | underline}}

.. currentmodule:: numpy

{% if objtype == 'property' %}
property
{% endif %}

.. auto{{ objtype }}:: numpy::ma.{{ objname }}

