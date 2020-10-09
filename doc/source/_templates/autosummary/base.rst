{% if objtype == 'property' %}
:orphan:
{% endif %}

{{ fullname | escape | underline}}

{% if objtype == 'property' %}
property
{% endif %}

.. auto{{ objtype }}:: {% block prefix %}{{ module }}{% endblock %}.{{ objname }}

