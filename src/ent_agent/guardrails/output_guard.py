"""Output guardrail (runs AFTER the model, BEFORE anything reaches a user or downstream system).

Never trust the model to self-censor (OWASP LLM05 improper output handling, LLM02, LLM07, LLM09).

  * system-prompt leak       -> canary token seen in output  => HARD FAIL (retry, then fallback)
  * secrets (keys/tokens)    -> HARD FAIL
  * PII                      -> redact and continue (a finance flow might choose to block instead)
  * active content           -> strip html tags + markdown images (classic data-exfil channel)
  * groundedness             -> answers built from retrieved docs must cite docs that were
                                actually retrieved this turn: `[doc:<id>]`
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .pii import SECRET_KINDS, find_pii, redact

CITATION_RE = re.compile(r"\[doc:([A-Za-z0-9_\-]+)\]")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML_TAG = re.compile(r"</?\s*[a-zA-Z][^>]{0,200}>")


@dataclass
class OutputVerdict:
    ok: bool
    text: str
    hard_fails: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def check_output(
    text: str,
    *,
    canary: str,
    allowed_doc_ids: set[str],
    require_citation: bool,
    max_chars: int,
) -> OutputVerdict:
    hard: list[str] = []
    notes: list[str] = []

    if canary and canary in text:
        hard.append("system_prompt_leak")

    if any(f.kind in SECRET_KINDS for f in find_pii(text)):
        hard.append("secret_in_output")

    cited = set(CITATION_RE.findall(text))
    if cited - allowed_doc_ids:
        hard.append("unverifiable_citation")
    if require_citation and not cited:
        hard.append("missing_citation")

    if hard:
        return OutputVerdict(False, text, hard, notes)

    cleaned = _MD_IMAGE.sub("[image removed]", text)
    cleaned = _HTML_TAG.sub("", cleaned)
    if cleaned != text:
        notes.append("active_content_stripped")

    cleaned, findings = redact(cleaned)
    if findings:
        notes.append("pii_redacted:" + ",".join(sorted({f.kind for f in findings})))

    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + " …[truncated]"
        notes.append("truncated")

    return OutputVerdict(True, cleaned, [], notes)
