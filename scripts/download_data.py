"""Download Microsoft Azure Predictive Maintenance sample CSVs into data/raw/."""

from __future__ import annotations

import sys
from pathlib import Path

import requests

# Official Microsoft Azure blob (may be unreachable in some networks/DNS)
AZURE_BLOB_BASE = "https://azuremlsampleexperiments.blob.core.windows.net/datasets"

# Verified working public mirror (same PdM_* filenames as Microsoft sample).
# Source: https://github.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy
# This repo hosts the original Microsoft Azure Predictive Maintenance CSVs used in
# Microsoft's predictive-maintenance tutorials (876k hourly telemetry rows, 2015).
GITHUB_MIRROR_BASE = (
    "https://raw.githubusercontent.com/ashishpatel26/"
    "Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy/master/data"
)

FILES = [
    "PdM_telemetry.csv",
    "PdM_errors.csv",
    "PdM_maint.csv",
    "PdM_failures.csv",
    "PdM_machines.csv",
]

# Expected minimum sizes (bytes) to reject truncated downloads
MIN_BYTES = {
    "PdM_telemetry.csv": 30_000_000,
    "PdM_errors.csv": 50_000,
    "PdM_maint.csv": 50_000,
    "PdM_failures.csv": 10_000,
    "PdM_machines.csv": 500,
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def download_file(url: str, dest: Path, timeout: int = 300) -> None:
    print(f"Downloading {url}")
    headers = {"User-Agent": "Failure-Sensor-Predictor/1.0 (dataset download)"}
    resp = requests.get(url, timeout=timeout, stream=True, headers=headers)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
    print(f"  -> saved {dest} ({dest.stat().st_size:,} bytes)")


def validate_file(name: str, dest: Path) -> None:
    min_size = MIN_BYTES.get(name, 100)
    if not dest.exists() or dest.stat().st_size < min_size:
        raise ValueError(
            f"{name} appears truncated ({dest.stat().st_size if dest.exists() else 0} bytes)"
        )


def main() -> int:
    raw_dir = project_root() / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        ("GitHub mirror (ashishpatel26)", GITHUB_MIRROR_BASE),
        ("Azure blob (official)", AZURE_BLOB_BASE),
    ]

    for name in FILES:
        dest = raw_dir / name
        if dest.exists() and dest.stat().st_size >= MIN_BYTES.get(name, 100):
            print(f"Already present: {dest} ({dest.stat().st_size:,} bytes)")
            continue

        last_err: Exception | None = None
        for label, base in sources:
            try:
                download_file(f"{base}/{name}", dest)
                validate_file(name, dest)
                print(f"  OK via {label}")
                break
            except Exception as err:
                last_err = err
                print(f"  {label} failed for {name}: {err}")
                if dest.exists():
                    dest.unlink()
        else:
            print(f"FAILED {name}: {last_err}", file=sys.stderr)
            return 1

    print("All dataset files ready in data/raw/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
