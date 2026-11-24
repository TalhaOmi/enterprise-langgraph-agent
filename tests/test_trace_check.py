"""The requirements checker is itself checked: a linter you have not tested can silently pass bad input."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("trace_check", ROOT / "scripts" / "trace_check.py")
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)


def run_with(tmp_path, monkeypatch, reqs):
    (tmp_path / "docs" / "se").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test_x.py").write_text("def test_real():\n    pass\n")
    (tmp_path / "docs" / "se" / "requirements.json").write_text(json.dumps(reqs))
    monkeypatch.setattr(tc, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["trace_check"])
    return tc.main()


GOOD = {
    "id": "REQ-001",
    "area": "a",
    "text": "The system shall do X.",
    "need": "n",
    "design": "d",
    "tests": ["tests/test_x.py::test_real"],
    "method": "Test",
}


def test_good_requirement_passes(tmp_path, monkeypatch):
    assert run_with(tmp_path, monkeypatch, [GOOD]) == 0


def test_vague_wording_and_missing_shall_fail(tmp_path, monkeypatch, capsys):
    assert run_with(tmp_path, monkeypatch, [{**GOOD, "text": "The agent should be safe"}]) == 1
    out = capsys.readouterr().out
    assert "no 'shall'" in out and "'safe'" in out and "'should'" in out


def test_missing_test_file_fails(tmp_path, monkeypatch, capsys):
    assert run_with(tmp_path, monkeypatch, [{**GOOD, "tests": ["tests/test_x.py::test_nope"]}]) == 1
    assert "test not found" in capsys.readouterr().out


def test_test_method_without_tests_fails_but_demonstration_is_allowed(tmp_path, monkeypatch):
    assert run_with(tmp_path, monkeypatch, [{**GOOD, "tests": []}]) == 1
    assert run_with(tmp_path, monkeypatch, [{**GOOD, "tests": [], "method": "Demonstration"}]) == 0


def test_duplicate_ids_fail(tmp_path, monkeypatch):
    assert run_with(tmp_path, monkeypatch, [GOOD, {**GOOD, "text": "The system shall do Y."}]) == 1
