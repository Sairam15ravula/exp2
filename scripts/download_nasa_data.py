"""Download NASA Li-ion Battery Aging Dataset.

Usage:
    python scripts/download_nasa_data.py

Downloads from NASA's Prognostic Data Repository.
Files are saved to data/raw/ (gitignored).

Note: NASA data is free for research. If download fails, visit:
https://www.nasa.gov/content/pcoe-data-set-repository
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

# NASA PCoE direct download URLs for the Li-ion battery aging dataset
NASA_BASE_URL = "https://data.nasa.gov/download"
BATTERY_FILES = {
    "B0005": f"{NASA_BASE_URL}/B0005.mat",
    "B0006": f"{NASA_BASE_URL}/B0006.mat",
    "B0007": f"{NASA_BASE_URL}/B0007.mat",
    "B0018": f"{NASA_BASE_URL}/B0018.mat",
}

OUTPUT_DIR = Path("data/raw")


def download_file(url: str, dest: Path) -> bool:
    """Download a single file. Returns True on success."""
    try:
        print(f"Downloading {dest.name}...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest.write_bytes(response.content)
        print(f"  -> Saved ({len(response.content)} bytes)")
        return True
    except Exception as e:
        print(f"  -> FAILED: {e}")
        return False


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    for name, url in BATTERY_FILES.items():
        dest = OUTPUT_DIR / f"{name}.mat"
        if dest.exists():
            print(f"{name}.mat already exists, skipping")
            success += 1
            continue
        if download_file(url, dest):
            success += 1

    print(f"\nDownloaded {success}/{len(BATTERY_FILES)} files to {OUTPUT_DIR}")
    if success < len(BATTERY_FILES):
        print("Some downloads failed. Visit the NASA PCoE repository manually:")
        print("https://www.nasa.gov/content/pcoe-data-set-repository")
        sys.exit(1)


if __name__ == "__main__":
    main()
