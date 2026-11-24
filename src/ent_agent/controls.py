"""Operational kill switch: stop all model/tool activity within one request, no redeploy.

Trip it with `touch .kill_switch` (or AGENT_KILL_SWITCH=true). See docs/RUNBOOK.md.
"""

from __future__ import annotations

from pathlib import Path

from .config import Settings


def kill_switch_active(settings: Settings) -> bool:
    return settings.kill_switch or Path(settings.kill_switch_file).exists()
