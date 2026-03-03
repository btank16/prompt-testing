import re
import json


def detect_variables(template: str) -> list:
    """Find {variable} placeholders in a template string.

    Only matches identifier-like names (letter/underscore start) to avoid
    matching JSON braces like {"type":
    """
    matches = re.findall(r'\{([a-zA-Z_]\w*)\}', template)
    seen = set()
    result = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def flatten_json(obj, parent_key='', sep='.'):
    """Recursively flatten nested dicts into dot-notation keys.

    {"demographics": {"population": 50000}} → {"demographics.population": 50000}
    Arrays of dicts use index notation (items.0.name).
    Primitive arrays kept as-is.
    """
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten_json(v, new_key, sep))
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                for i, item in enumerate(v):
                    items.update(flatten_json(item, f"{new_key}{sep}{i}", sep))
            else:
                items[new_key] = v
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            if isinstance(item, dict):
                items.update(flatten_json(item, new_key, sep))
            else:
                items[new_key] = item
    else:
        items[parent_key] = obj
    return items


class _SafeDict(dict):
    """Dict subclass that leaves missing keys as {key} in format_map."""
    def __missing__(self, key):
        return '{' + key + '}'


def substitute_variables(template: str, variables: dict) -> str:
    """Substitute {variable} placeholders, leaving unmatched ones intact."""
    return template.format_map(_SafeDict(variables))


def extract_schema_from_stored_format(stored_format) -> dict:
    """Convert from existing Perplexity/OpenAI schema wrapper format to raw JSON schema.

    Handles:
      - String input (parses JSON first)
      - {"type": "json_schema", "json_schema": {"schema": {...}}} (Perplexity format)
      - {"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}} (OpenAI format)
      - Raw schema dict (returned as-is)
    """
    if isinstance(stored_format, str):
        try:
            stored_format = json.loads(stored_format)
        except json.JSONDecodeError:
            return {}

    if not isinstance(stored_format, dict):
        return {}

    # Nested wrapper: {"type": "json_schema", "json_schema": {"schema": {...}}}
    if "json_schema" in stored_format:
        inner = stored_format["json_schema"]
        if isinstance(inner, dict) and "schema" in inner:
            return inner["schema"]
        return inner

    # Already a raw schema
    if "type" in stored_format and "properties" in stored_format:
        return stored_format

    return stored_format


def expand_json_to_rows(parsed):
    """Expand a parsed JSON response into multiple rows when it contains parallel arrays.

    Scalars are repeated on every row.  Parallel arrays are zipped so that
    index-aligned items share a row (array1[0] with array2[0], etc.).
    Arrays of objects are flattened with the array key as prefix.

    Returns a list of flat dicts — one per output row.
    """
    # Top-level list: each item becomes a row
    if isinstance(parsed, list):
        rows = []
        for item in parsed:
            if isinstance(item, dict):
                rows.append(flatten_json(item))
            else:
                rows.append({"_value": item})
        return rows if rows else [{}]

    if not isinstance(parsed, dict):
        return [{"_raw_response": str(parsed)}]

    scalars = {}
    arrays = {}

    for k, v in parsed.items():
        if isinstance(v, list):
            arrays[k] = v
        elif isinstance(v, dict):
            scalars.update(flatten_json(v, parent_key=k))
        else:
            scalars[k] = v

    # No arrays — single row with just scalars
    if not arrays:
        return [scalars]

    max_len = max(len(arr) for arr in arrays.values())
    if max_len == 0:
        return [scalars]

    rows = []
    for i in range(max_len):
        row = dict(scalars)
        for arr_key, arr_val in arrays.items():
            if i < len(arr_val):
                item = arr_val[i]
                if isinstance(item, dict):
                    row.update(flatten_json(item, parent_key=arr_key))
                else:
                    row[arr_key] = item
            else:
                row[arr_key] = None
        rows.append(row)

    return rows


def normalize_response(provider: str, raw: dict) -> dict:
    """Convert raw API responses into a standard format.

    Returns: {"content": str, "usage": {...}, "model": str, "citations": list|None, "search_results": list|None}
    """
    if provider == "gemini":
        return raw  # already normalized by GeminiClient

    if provider == "perplexity":
        content = ""
        if "choices" in raw and raw["choices"]:
            content = raw["choices"][0].get("message", {}).get("content", "")
        usage_raw = raw.get("usage", {})
        return {
            "content": content,
            "usage": {
                "prompt_tokens": usage_raw.get("prompt_tokens", 0),
                "completion_tokens": usage_raw.get("completion_tokens", 0),
                "total_tokens": usage_raw.get("total_tokens", 0),
            },
            "model": raw.get("model", ""),
            "citations": raw.get("citations"),
            "search_results": raw.get("search_results"),
        }

    if provider == "openai":
        content = ""
        if "choices" in raw and raw["choices"]:
            content = raw["choices"][0].get("message", {}).get("content", "")
        usage_raw = raw.get("usage", {})
        return {
            "content": content,
            "usage": {
                "prompt_tokens": usage_raw.get("prompt_tokens", 0),
                "completion_tokens": usage_raw.get("completion_tokens", 0),
                "total_tokens": usage_raw.get("total_tokens", 0),
            },
            "model": raw.get("model", ""),
            "citations": None,
            "search_results": None,
        }

    return raw


def detect_provider_from_model(model_name: str) -> str:
    """Return 'perplexity', 'openai', or 'gemini' based on model name prefix."""
    m = model_name.lower()
    if m.startswith("sonar"):
        return "perplexity"
    if m.startswith("gpt"):
        return "openai"
    if m.startswith("gemini"):
        return "gemini"
    return "unknown"


def build_response_format(provider: str, raw_schema: dict) -> dict:
    """Wrap a raw JSON schema into the provider-specific envelope."""
    if provider == "perplexity":
        return {
            "type": "json_schema",
            "json_schema": {"schema": raw_schema},
        }
    if provider == "openai":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                **raw_schema,
            },
        }
    # gemini — client handles wrapping
    return raw_schema
