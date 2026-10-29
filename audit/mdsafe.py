"""Markdown helpers for Streamlit: a bare ``$`` starts LaTeX math in st.markdown / st.caption, so money
like "$0.003 ... $0" renders as an italic formula. :func:`escape_dollars` escapes every ``$`` outside code
spans and fenced code blocks (where it is shown literally anyway)."""
from __future__ import annotations

import re

_FENCE = re.compile(r"^(\s*)(```|~~~)")
_SPAN = re.compile(r"(`+)(.+?)\1", re.S)


def _escape_text(s: str) -> str:
    return re.sub(r"(?<!\\)\$", r"\\$", s)


def _escape_line(line: str) -> str:
    out, pos = [], 0
    for m in _SPAN.finditer(line):
        out.append(_escape_text(line[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(_escape_text(line[pos:]))
    return "".join(out)


def escape_dollars(text: str) -> str:
    """Escape ``$`` for Streamlit markdown, leaving code spans and fenced blocks untouched."""
    if not text or "$" not in text:
        return text
    out, fenced = [], False
    for line in text.split("\n"):
        if _FENCE.match(line):
            fenced = not fenced
            out.append(line)
        elif fenced:
            out.append(line)
        else:
            out.append(_escape_line(line))
    return "\n".join(out)
