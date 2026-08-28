from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_parse() -> None:
    failures: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc}")
    assert not failures, "\n".join(failures)


def test_json_configurations_parse() -> None:
    failures: list[str] = []
    for path in sorted((ROOT / "config").glob("*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            failures.append(f"{path.name}: {exc}")
    assert not failures, "\n".join(failures)


def test_no_machine_specific_user_path_in_tracked_text() -> None:
    failures: list[str] = []
    for suffix in ("*.py", "*.json", "*.md"):
        for path in ROOT.rglob(suffix):
            if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if "c:\\users\\xg625" in text:
                failures.append(str(path.relative_to(ROOT)))
    assert not failures, "Machine-specific paths: " + ", ".join(failures)


if __name__ == "__main__":
    test_python_sources_parse()
    test_json_configurations_parse()
    test_no_machine_specific_user_path_in_tracked_text()
    print("repository integrity checks passed")
