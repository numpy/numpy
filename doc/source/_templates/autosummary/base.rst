{% if objtype == 'property' %}
:orphan:
{% endif %}

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{% if objtype == 'property' %}
property
{% endif %}

.. auto{{ objtype }}:: {{ objname }}

