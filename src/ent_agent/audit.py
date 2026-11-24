"""Tamper-evident audit trail (EU AI Act Art. 12 record-keeping, SOC 2, GDPR accountability).

* Append-only JSONL; every entry embeds the hash of the previous entry (hash chain).
* Stores fingerprints of user content, never raw prompts/PII.
* In production ship this to your SIEM / WORM storage; the chain lets you PROVE nothing was edited.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("ent_agent.audit")
GENESIS = "0" * 64


def fingerprint(text: str) -> str:
    """Short stable hash so events are correlatable without storing content."""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _digest(entry: dict[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._seq = 0
        self._prev = GENESIS
        if self.path.exists():
            last = None
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        last = json.loads(line)
            if last:
                self._seq, self._prev = last["seq"], last["hash"]

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            entry = {"seq": self._seq, "ts": time.time(), "event": event, **fields, "prev": self._prev}
            entry["hash"] = _digest(entry)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
            self._prev = entry["hash"]
        log.info("audit %s", event, extra={"audit": entry})
        return entry

    @staticmethod
    def verify(path: str | Path) -> tuple[bool, int | None]:
        """Return (ok, first_bad_seq). Detects edits, deletions and reordering."""
        prev = GENESIS
        expected_seq = 1
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("prev") != prev or entry.get("seq") != expected_seq:
                return False, entry.get("seq")
            if _digest(entry) != entry.get("hash"):
                return False, entry.get("seq")
            prev, expected_seq = entry["hash"], expected_seq + 1
        return True, None
