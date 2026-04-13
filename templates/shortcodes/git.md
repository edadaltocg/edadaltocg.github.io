<details>
  <summary>Code</summary>

```{% if language %}{{language}}{% endif %},linenos,hide_lines={{hl}}
{% set data = load_data(url=url, format="plain") %}
{{ data | safe  }}
```

Source: [`{{url}}`]({{url}}).

</details>
