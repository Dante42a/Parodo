"""
Markdown-сводка результатов pytest для страницы запуска GitHub Actions.

Usage:
    pytest --junitxml=pytest-report.xml
    python scripts/test_summary.py pytest-report.xml >> "$GITHUB_STEP_SUMMARY"
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ICONS = {"passed": "✅", "failed": "❌", "error": "💥", "skipped": "⏭️"}


def outcome(case: ET.Element) -> str:
    for tag in ("failure", "error", "skipped"):
        if case.find(tag) is not None:
            return {"failure": "failed"}.get(tag, tag)
    return "passed"


def main():
    report = Path(sys.argv[1])
    if not report.is_file():
        print("### Тесты\n\nОтчёт pytest не создан — тесты не запустились (см. лог шага).")
        return
    cases = list(ET.parse(report).getroot().iter("testcase"))
    results = [(c.get("classname", ""), c.get("name", ""), outcome(c)) for c in cases]
    counts = {k: sum(r[2] == k for r in results) for k in ICONS}

    print("### Тесты\n")
    print(" · ".join(f"{ICONS[k]} {k}: {v}" for k, v in counts.items() if v) + "\n")
    print("| | Файл | Тест |\n|---|---|---|")
    for classname, name, result in sorted(results, key=lambda r: (r[2] == "passed", r[0], r[1])):
        print(f"| {ICONS[result]} | `{classname.removeprefix('tests.')}` | `{name}` |")


if __name__ == "__main__":
    main()
