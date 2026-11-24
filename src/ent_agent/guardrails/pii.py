"""Deterministic PII / secret detection. Fast, free, and runs on EVERY call (input and output).

Regex is a baseline, not a ceiling: in production add a model-based detector (e.g. Microsoft
Presidio, LLM Guard) behind the same `redact` interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SECRET_KINDS = frozenset({"aws_key", "api_key", "bearer", "jwt"})


@dataclass(frozen=True)
class Finding:
    kind: str
    start: int
    end: int


_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("api_key", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_\-]{16,}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("sin", re.compile(r"\b\d{3}[ -]\d{3}[ -]\d{3}\b")),
    ("phone", re.compile(r"(?<!\w)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b")),
]
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def find_pii(text: str) -> list[Finding]:
    spans: list[Finding] = []
    for kind, pattern in _PATTERNS:
        spans += [Finding(kind, m.start(), m.end()) for m in pattern.finditer(text)]
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            spans.append(Finding("credit_card", m.start(), m.end()))
    # earliest first, longest first; drop overlaps
    spans.sort(key=lambda f: (f.start, -(f.end - f.start)))
    kept: list[Finding] = []
    for f in spans:
        if not kept or f.start >= kept[-1].end:
            kept.append(f)
    return kept


def redact(text: str) -> tuple[str, list[Finding]]:
    findings = find_pii(text)
    out = text
    for f in reversed(findings):
        out = out[: f.start] + f"[REDACTED:{f.kind}]" + out[f.end :]
    return out, findings
