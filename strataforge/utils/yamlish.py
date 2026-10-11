"""Minimal YAML-subset parser (zero-dependency fallback).

Supports the small YAML dialect this project actually uses: nested mappings via
indentation, block lists (`- item`), inline lists (`[a, b, c]`), and inline
comments (`# ...`). Full enough for config files, small enough to audit.

Used automatically by `Config.from_dict` when PyYAML is not installed (the
managed Python 3.13 runtime has no PyYAML).
"""


def _strip_comment(line):
    """Remove a trailing ` # comment`, respecting a leading `#` on its own."""
    s = line
    if "#" in s:
        idx = s.index("#")
        s = s[:idx]
    return s.rstrip()


def parse(text):
    """Parse a YAML-subset document into nested dict/list structures."""
    lines = []
    for raw in text.splitlines():
        if raw.strip() == "":
            continue
        if raw.strip().startswith("#"):
            continue
        lines.append(raw.rstrip("\n"))

    root = {}
    # stack entries: [indent, container, key]  (container is dict or list)
    stack = [[-1, root, None]]

    def _indent_of(line):
        return len(line) - len(line.lstrip(" "))

    for line in lines:
        indent = _indent_of(line)
        body = _strip_comment(line.strip())
        if body == "":
            continue

        # pop stack until the current indent is deeper than the top
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            stack = [[-1, root, None]]

        container = stack[-1][1]
        parent_key = stack[-1][2]

        if body.startswith("- "):
            item_text = body[2:].strip()
            if isinstance(container, dict):
                # a block list arrived under a mapping key -> convert placeholder
                lst = []
                if parent_key is not None and stack[-2][1] is not None:
                    stack[-2][1][parent_key] = lst
                container = lst
                stack[-1][1] = lst
            if ":" in item_text:
                k, _, v = item_text.partition(":")
                k = k.strip()
                v = v.strip()
                d = {}
                container.append(d)
                stack.append([indent, d, k])
                if v != "":
                    d[k] = _coerce(v)
                else:
                    d[k] = None
            else:
                container.append(_coerce(item_text))
            continue

        if ":" in body:
            k, _, v = body.partition(":")
            k = k.strip()
            v = v.strip()
            if v == "":
                # nested mapping or list follows
                child = {}
                if isinstance(container, list):
                    container.append(child)
                else:
                    container[k] = child
                stack.append([indent, child, k])
            else:
                val = _coerce(v)
                if isinstance(container, list):
                    container.append({k: val})
                else:
                    container[k] = val

    return root


def _coerce(v):
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if inner == "":
            return []
        return [_coerce(x.strip()) for x in inner.split(",")]
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    if v in ("null", "~", ""):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    return v
