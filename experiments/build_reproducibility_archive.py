"""
Build Clean Standalone Reproducibility Archive
==============================================
Packs all source code, 102 tests, experiments, results, and documentation
into CancelableFingerprintHybrid.zip, excluding dataset/ images and cache files.
"""

import os
import zipfile
from pathlib import Path

BASE_DIR = Path(r"C:\Users\mella\.gemini\antigravity\scratch\CancelableFingerprintHybrid")
OUTPUT_ZIP = BASE_DIR / "CancelableFingerprintHybrid.zip"

EXCLUDE_DIRS = {
    "dataset",
    ".pytest_cache",
    "__pycache__",
    ".git"
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo"
}

EXCLUDE_FILES = {
    "CancelableFingerprintHybrid.zip"
}

def build_zip():
    print(f"Building clean reproducibility archive: {OUTPUT_ZIP}")
    total_files = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(BASE_DIR):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
            
            for file in sorted(files):
                if file in EXCLUDE_FILES:
                    continue
                ext = os.path.splitext(file)[1].lower()
                if ext in EXCLUDE_EXTENSIONS:
                    continue
                
                full_path = Path(root) / file
                rel_path = full_path.relative_to(BASE_DIR)
                
                # Check if path contains excluded parts
                parts = rel_path.parts
                if any(p in EXCLUDE_DIRS for p in parts):
                    continue
                
                zf.write(full_path, arcname=str(rel_path))
                total_files += 1
                total_uncompressed_bytes += full_path.stat().st_size

    archive_size = OUTPUT_ZIP.stat().st_size
    print(f"Successfully created archive:")
    print(f"  Files packed: {total_files}")
    print(f"  Uncompressed size: {total_uncompressed_bytes:,} bytes ({total_uncompressed_bytes / 1024 / 1024:.2f} MB)")
    print(f"  Compressed size:   {archive_size:,} bytes ({archive_size / 1024:.2f} KB)")

if __name__ == "__main__":
    build_zip()
