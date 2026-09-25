{% for section, _ in sections.items() %}
{% if section %}
## {{ section }}

{% endif %}
{% if sections[section] %}
{% for category, val in definitions.items() if category in sections[section] %}

{% if section %}###{% else %}##{% endif %} {{ definitions[category]['name'] }}

{% if definitions[category]['showcontent'] %}
{% for text, values in sections[section][category].items() %}
{{ text }}

{{ get_indent(text) }}({{ values|join(', ') }})

{% endfor %}
{% else %}
- {{ sections[section][category]['']|join(', ') }}

{% endif %}
{% if sections[section][category]|length == 0 %}
No significant changes.

{% endif %}
{% endfor %}
{% else %}
(no release note snippets found)

{% endif %}
{% endfor %}
