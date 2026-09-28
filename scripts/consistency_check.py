"""Consistency check: README, metrics.json, Streamlit app."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    data_readme = (ROOT / "data" / "README.md").read_text(encoding="utf-8")
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    test = metrics["metrics_test"]
    issues: list[str] = []

    for name, text in [
        ("README.md", readme),
        ("data/README.md", data_readme),
        ("app.py", app),
    ]:
        if re.search(r"(?i)simulated data", text):
            issues.append(f'{name}: still contains "Simulated data"')

    expected = {
        "precision": round(test["precision"], 3),
        "recall": round(test["recall"], 3),
        "f1": round(test["f1"], 3),
        "accuracy": round(test["accuracy"], 3),
    }
    for key, value in expected.items():
        if f"{value:.3f}" not in readme:
            issues.append(f"README missing {key}={value:.3f}")

    n_rows = metrics["dataset"]["n_rows"]
    if f"{n_rows:,}" not in readme and str(n_rows) not in readme:
        issues.append(f"README missing n_rows={n_rows}")

    if metrics["model"] not in readme:
        issues.append("README model name mismatch")

    if metrics["horizon_hours"] != 24:
        issues.append("metrics horizon != 24")
    if metrics["rolling_window_hours"] != 12:
        issues.append("metrics rolling != 12")

    if "Microsoft Azure PdM" not in app:
        issues.append("app dataset label mismatch")
    if "METRICS_PATH" not in app:
        issues.append("app does not reference metrics path")
    if "approximates 12h rolling" not in app:
        issues.append("app missing single-row approximation note")
    if "chronological test set" not in app:
        issues.append("app demo wording unclear")

    themes = [
        "Historical benchmark",
        "not live production",
        "precision, recall, and F1",
        "error_count_24h",
        "CSV upload",
    ]
    for phrase in themes:
        if phrase not in readme:
            issues.append(f"README missing limitation theme: {phrase}")

    print("=== Consistency check ===")
    print(
        "metrics_test:",
        {k: round(test[k], 6) for k in ["precision", "recall", "f1", "accuracy"]},
    )
    print("dataset n_rows:", n_rows)
    print("model:", metrics["model"])
    print(
        "horizon:",
        metrics["horizon_hours"],
        "rolling:",
        metrics["rolling_window_hours"],
    )
    print("simulated mentions in README:", len(re.findall(r"(?i)simulat", readme)))
    print(
        "simulated mentions in data/README:",
        len(re.findall(r"(?i)simulat", data_readme)),
    )
    print("simulated mentions in app.py:", len(re.findall(r"(?i)simulat", app)))

    if issues:
        print("ISSUES:")
        for item in issues:
            print(" -", item)
        return 1

    print("PASS: README, metrics.json, and app.py are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
