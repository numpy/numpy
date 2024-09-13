{% extends "!autosummary/module.rst" %}

{# This file is almost the same as the default, but adds :toctree: to the autosummary directives.
   The original can be found at `sphinx/ext/autosummary/templates/autosummary/module.rst`. #}

{% block attributes %}
{% if attributes %}
   .. rubric:: Module Attributes

   .. autosummary::
      :toctree:
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
{% endif %}
{% endblock %}

{% block functions %}
{% if functions %}
   .. rubric:: Functions

   .. autosummary::
      :toctree:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
{% endif %}
{% endblock %}

{% block classes %}
{% if classes %}
   .. rubric:: Classes

   .. autosummary::
      :toctree:
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
{% endif %}
{% endblock %}
