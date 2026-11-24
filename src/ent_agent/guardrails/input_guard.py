"""Input guardrail (runs BEFORE the model sees anything). OWASP LLM01 / LLM02 / LLM10.

Layers, cheapest first:
  1. normalise (NFKC, strip zero-width/control chars)  -> defeats simple obfuscation
  2. hard length cap                                   -> cost + stuffing attacks
  3. injection heuristics                              -> block, return a RISK LABEL only
  4. PII / secret redaction                            -> data never leaves your boundary

Heuristics are one layer of defence-in-depth, never the only one: least-privilege tools,
HITL approval and output checks assume this layer WILL be bypassed sometimes.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .pii import redact

INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "override_instructions",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|system|all|any)\b"
            r".{0,30}\b(instructions?|prompts?|rules?|guidelines?|messages?)\b"
        ),
    ),
    (
        "prompt_extraction",
        re.compile(
            r"\b(reveal|show|print|repeat|output|leak|display|tell me|what (is|are))\b.{0,25}"
            r"\b(your|the)\b.{0,15}\b(system|hidden|initial|original|internal)\b.{0,10}\b(prompt|instructions?|message|rules)\b"
        ),
    ),
    ("persona_jailbreak", re.compile(r"\b(developer mode|do anything now|jailbreak|\bdan\b mode|unrestricted mode)\b")),
    ("persona_jailbreak", re.compile(r"\byou are now\b.{0,30}\b(unrestricted|unfiltered|dan|free of)\b")),
    (
        "disable_safety",
        re.compile(
            r"\b(bypass|disable|turn off|override|circumvent)\b.{0,20}\b(safety|guardrails?|filters?|polic(y|ies)|restrictions?)\b"
        ),
    ),
    ("fake_role_tags", re.compile(r"</?\s*(system|assistant|tool|developer)\s*>|\[/?(inst|system)\]")),
]

_ZERO_WIDTH_AND_CONTROL = re.compile(r"[​-‏‪-‮⁠-⁤﻿]|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class InputVerdict:
    allowed: bool
    text: str  # normalised + redacted text, safe to send to the model
    reasons: list[str] = field(default_factory=list)
    pii_kinds: list[str] = field(default_factory=list)


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_AND_CONTROL.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def injection_labels(text: str) -> list[str]:
    lowered = text.lower()
    return sorted({label for label, pat in INJECTION_PATTERNS if pat.search(lowered)})


def check_input(raw: str, *, max_chars: int) -> InputVerdict:
    if len(raw) > max_chars:
        return InputVerdict(False, "", ["input_too_long"])
    text = normalise(raw)
    if not text:
        return InputVerdict(False, "", ["empty_input"])
    labels = injection_labels(text)
    if labels:
        return InputVerdict(False, "", [f"injection:{label}" for label in labels])
    redacted, findings = redact(text)
    return InputVerdict(True, redacted, [], sorted({f.kind for f in findings}))
