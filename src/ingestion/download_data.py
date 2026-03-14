"""
Download Cricsheet datasets (JSON format) and People Register.

Usage:
    python src/ingestion/download_data.py                  # Download all datasets
    python src/ingestion/download_data.py --datasets t20i ipl  # Download specific ones
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import CRICSHEET_URLS, DATASET_DIRS, RAW_REGISTER_DIR


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> Path:
    """Download a file from a URL with progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    filepath = dest_path / filename

    print(f"\n📥 Downloading: {url}")
    print(f"   → {filepath}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(filepath, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"   ✅ Downloaded: {filepath.name} ({filepath.stat().st_size / 1024 / 1024:.1f} MB)")
    return filepath


def extract_zip(zip_path: Path, extract_to: Path) -> int:
    """Extract a zip file and return the count of extracted files."""
    print(f"📦 Extracting: {zip_path.name} → {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        json_files = [m for m in members if m.endswith(".json")]
        zf.extractall(extract_to)

    print(f"   ✅ Extracted {len(json_files)} JSON files")
    return len(json_files)


def download_people_register() -> Path:
    """Download the Cricsheet People Register CSV."""
    url = CRICSHEET_URLS["people_register"]
    RAW_REGISTER_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RAW_REGISTER_DIR / "people.csv"

    print(f"\n📥 Downloading People Register: {url}")
    response = requests.get(url)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        f.write(response.content)

    # Count rows
    line_count = response.text.count("\n")
    print(f"   ✅ Saved: {filepath.name} ({line_count} entries)")
    return filepath


def download_dataset(dataset_key: str) -> dict:
    """Download and extract a single dataset. Returns stats dict."""
    if dataset_key == "people_register":
        filepath = download_people_register()
        return {"dataset": dataset_key, "file": str(filepath), "match_count": 0}

    url = CRICSHEET_URLS[dataset_key]
    dest_dir = DATASET_DIRS[dataset_key]
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Download zip
    zip_path = download_file(url, dest_dir.parent)

    # Extract
    match_count = extract_zip(zip_path, dest_dir)

    # Clean up zip
    zip_path.unlink()
    print(f"   🗑️  Removed zip: {zip_path.name}")

    return {"dataset": dataset_key, "dir": str(dest_dir), "match_count": match_count}


def main():
    parser = argparse.ArgumentParser(description="Download Cricsheet datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(CRICSHEET_URLS.keys()),
        default=list(CRICSHEET_URLS.keys()),
        help="Datasets to download (default: all)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🏏 CRICSHEET DATA DOWNLOADER")
    print("=" * 60)
    print(f"Datasets to download: {', '.join(args.datasets)}")

    results = []
    for dataset_key in args.datasets:
        try:
            result = download_dataset(dataset_key)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Failed to download {dataset_key}: {e}")
            results.append({"dataset": dataset_key, "error": str(e)})

    # Summary
    print("\n" + "=" * 60)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    total_matches = 0
    for r in results:
        if "error" in r:
            print(f"  ❌ {r['dataset']}: FAILED - {r['error']}")
        else:
            count = r.get("match_count", 0)
            total_matches += count
            if count > 0:
                print(f"  ✅ {r['dataset']}: {count} match files")
            else:
                print(f"  ✅ {r['dataset']}: downloaded")

    print(f"\n  Total match files: {total_matches}")
    print("=" * 60)


if __name__ == "__main__":
    main()
