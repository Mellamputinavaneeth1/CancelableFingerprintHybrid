"""
SOCOFing Manifest Builder & Identity Resolution Service (Phases 0, 1, 2).

Discovers, audits, and parses fingerprint images in the SOCOFing hierarchy:
- Verifies file integrity, computes SHA-256 digests, and image dimensions.
- Normalizes filename metadata (Subject, Sex, Hand, Finger, Alteration, Severity).
- Constructs canonical biometric fingerprint identities: fingerprint_id = subject_id|hand|finger.
- Resolves clean Real gallery references for every altered probe.
- Emits archive_audit.json, file_manifest.csv, file_hashes.csv, metadata_manifest.csv,
  and unmapped_files.csv, satisfying Gates 0, 1, and 2.
"""

import os
import re
import csv
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import cv2

SOCOFING_DPI = 500.0

# Canonical finger names allowed
VALID_FINGERS = {"thumb", "index", "middle", "ring", "little"}
VALID_HANDS = {"left", "right"}
VALID_SEVERITIES = {"real", "easy", "medium", "hard"}


def parse_socofing_filename(filename: str, parent_dir: str = "") -> Optional[Dict[str, Any]]:
    """
    Parses a SOCOFing filename into structured metadata fields.
    
    Expected patterns:
      Real:    {id}__{gender}_{hand}_{finger}_finger.BMP
      Altered: {id}__{gender}_{hand}_{finger}_finger_{alteration_code}.BMP
      
    Examples:
      1__M_Left_index_finger.BMP
      10__F_Right_thumb_finger_CR.BMP
      005__M_Right_little_finger_Zcut.BMP
      7__F_Left_middle_finger_Obl.BMP
    """
    basename, ext = os.path.splitext(filename)
    if ext.lower() not in {".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg"}:
        return None

    # Altered pattern
    pattern_altered = r"^(\d+)__([MFmf])_([A-Za-z]+)_([A-Za-z]+)_finger_([A-Za-z0-9_]+)$"
    m_alt = re.match(pattern_altered, basename)
    if m_alt:
        subj_num = int(m_alt.group(1))
        sex = m_alt.group(2).upper()
        hand = m_alt.group(3).capitalize()
        finger = m_alt.group(4).lower()
        alt_code = m_alt.group(5)

        # Validate hand & finger
        if hand.lower() not in VALID_HANDS or finger not in VALID_FINGERS:
            return None

        # Canonical alteration class mapping
        alt_code_lower = alt_code.lower()
        if "obl" in alt_code_lower or "obliteration" in alt_code_lower:
            alt_type = "Obliteration"
        elif "cr" in alt_code_lower or "rotation" in alt_code_lower:
            alt_type = "Central_Rotation"
        elif "zcut" in alt_code_lower or "z_cut" in alt_code_lower or "z" in alt_code_lower:
            alt_type = "Z-cut"
        else:
            alt_type = alt_code

        # Severity inferred from parent_dir or filename
        parent_lower = parent_dir.lower()
        if "easy" in parent_lower:
            severity = "Easy"
        elif "medium" in parent_lower:
            severity = "Medium"
        elif "hard" in parent_lower:
            severity = "Hard"
        else:
            severity = "Unknown"

        fingerprint_id = f"{subj_num:03d}|{hand}|{finger}"
        return {
            "subject_id": subj_num,
            "sex_metadata": sex,
            "hand": hand,
            "finger": finger,
            "fingerprint_id": fingerprint_id,
            "source_fingerprint_id": fingerprint_id,
            "is_altered": True,
            "alteration_type": alt_type,
            "alteration_code_raw": alt_code,
            "severity": severity,
            "partition": f"Altered-{severity}" if severity != "Unknown" else "Altered",
            "is_gallery_candidate": False,
            "is_probe_candidate": True,
            "parse_status": "VALID_ALTERED"
        }

    # Real pattern
    pattern_real = r"^(\d+)__([MFmf])_([A-Za-z]+)_([A-Za-z]+)_finger$"
    m_real = re.match(pattern_real, basename)
    if m_real:
        subj_num = int(m_real.group(1))
        sex = m_real.group(2).upper()
        hand = m_real.group(3).capitalize()
        finger = m_real.group(4).lower()

        if hand.lower() not in VALID_HANDS or finger not in VALID_FINGERS:
            return None

        fingerprint_id = f"{subj_num:03d}|{hand}|{finger}"
        return {
            "subject_id": subj_num,
            "sex_metadata": sex,
            "hand": hand,
            "finger": finger,
            "fingerprint_id": fingerprint_id,
            "source_fingerprint_id": fingerprint_id,
            "is_altered": False,
            "alteration_type": "None",
            "alteration_code_raw": "None",
            "severity": "Real",
            "partition": "Real",
            "is_gallery_candidate": True,
            "is_probe_candidate": False,
            "parse_status": "VALID_REAL"
        }

    return None


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_and_build_manifest(dataset_root: str,
                             output_dir: str = "artifacts/socofing") -> Dict[str, Any]:
    """
    Performs full archive audit (Gate 0), parses metadata (Gate 1), and resolves
    identities and gallery linkages (Gate 2).
    """
    os.makedirs(output_dir, exist_ok=True)

    file_records: List[Dict[str, Any]] = []
    excluded_records: List[Dict[str, Any]] = []
    unmapped_records: List[Dict[str, Any]] = []

    seen_hashes: Dict[str, str] = {}
    gallery_by_id: Dict[str, Dict[str, Any]] = {}

    file_counter = 0

    # 1. Discover and parse all files
    for root, dirs, files in os.walk(dataset_root):
        rel_dir = os.path.relpath(root, dataset_root)
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            rel_path = os.path.normpath(os.path.join(rel_dir, fname))
            file_counter += 1
            file_id = f"file_{file_counter:06d}"

            # Check if file is readable
            if not os.path.isfile(fpath):
                excluded_records.append({
                    "file_id": file_id,
                    "relative_path": rel_path,
                    "reason": "NOT_A_FILE"
                })
                continue

            try:
                byte_size = os.path.getsize(fpath)
                sha256_digest = compute_file_sha256(fpath)
            except Exception as e:
                excluded_records.append({
                    "file_id": file_id,
                    "relative_path": rel_path,
                    "reason": f"READ_ERROR: {str(e)}"
                })
                continue

            # Duplicate hash check
            is_duplicate = sha256_digest in seen_hashes
            if not is_duplicate:
                seen_hashes[sha256_digest] = rel_path

            # Image decoding check
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                excluded_records.append({
                    "file_id": file_id,
                    "relative_path": rel_path,
                    "byte_size": byte_size,
                    "sha256": sha256_digest,
                    "reason": "IMAGE_DECODE_FAILED"
                })
                continue

            height, width = img.shape

            # Parse filename
            parsed = parse_socofing_filename(fname, parent_dir=rel_dir)
            if not parsed:
                unmapped_records.append({
                    "file_id": file_id,
                    "relative_path": rel_path,
                    "filename": fname,
                    "byte_size": byte_size,
                    "sha256": sha256_digest,
                    "width": width,
                    "height": height,
                    "reason": "UNPARSEABLE_FILENAME"
                })
                continue

            record = {
                "file_id": file_id,
                "relative_path": rel_path,
                "absolute_path": os.path.abspath(fpath),
                "filename": fname,
                "sha256": sha256_digest,
                "byte_size": byte_size,
                "width": width,
                "height": height,
                "is_duplicate_pixel_hash": is_duplicate,
                **parsed
            }

            file_records.append(record)

            if record["is_gallery_candidate"]:
                f_id = record["fingerprint_id"]
                if f_id not in gallery_by_id:
                    gallery_by_id[f_id] = record

    # 2. Link altered probes to clean Real gallery images (Gate 2)
    missing_gallery_links = 0
    linked_probes = 0

    for rec in file_records:
        if rec["is_probe_candidate"]:
            f_id = rec["source_fingerprint_id"]
            if f_id in gallery_by_id:
                rec["has_clean_gallery"] = True
                rec["gallery_file_id"] = gallery_by_id[f_id]["file_id"]
                linked_probes += 1
            else:
                rec["has_clean_gallery"] = False
                rec["gallery_file_id"] = None
                missing_gallery_links += 1

    # 3. Aggregate audit metrics
    unique_subjects = sorted(list({r["subject_id"] for r in file_records}))
    unique_fingerprints = sorted(list({r["fingerprint_id"] for r in file_records}))
    real_count = sum(1 for r in file_records if r["partition"] == "Real")
    altered_easy = sum(1 for r in file_records if r["severity"] == "Easy")
    altered_medium = sum(1 for r in file_records if r["severity"] == "Medium")
    altered_hard = sum(1 for r in file_records if r["severity"] == "Hard")

    audit_summary = {
        "dataset_root": os.path.abspath(dataset_root),
        "total_files_discovered": file_counter,
        "valid_manifest_records": len(file_records),
        "excluded_files_count": len(excluded_records),
        "unmapped_files_count": len(unmapped_records),
        "unique_subjects_count": len(unique_subjects),
        "unique_fingerprints_count": len(unique_fingerprints),
        "real_gallery_images": real_count,
        "altered_probe_images": len(file_records) - real_count,
        "linked_probes": linked_probes,
        "missing_gallery_links": missing_gallery_links,
        "severity_counts": {
            "Real": real_count,
            "Altered-Easy": altered_easy,
            "Altered-Medium": altered_medium,
            "Altered-Hard": altered_hard
        },
        "gates": {
            "Gate_0_Archive": len(excluded_records) == 0 and file_counter > 0,
            "Gate_1_Parser": len(unmapped_records) == 0 and len(file_records) > 0,
            "Gate_2_Identity_Linkage": missing_gallery_links == 0
        }
    }

    # 4. Save CSV and JSON outputs
    audit_json_path = os.path.join(output_dir, "archive_audit.json")
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    # Save metadata_manifest.csv
    if file_records:
        manifest_csv_path = os.path.join(output_dir, "metadata_manifest.csv")
        keys = list(file_records[0].keys())
        with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(file_records)

    # Save file_hashes.csv
    hashes_csv_path = os.path.join(output_dir, "file_hashes.csv")
    with open(hashes_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_id", "relative_path", "sha256", "byte_size"])
        for r in file_records:
            writer.writerow([r["file_id"], r["relative_path"], r["sha256"], r["byte_size"]])

    # Save unmapped / excluded CSVs
    if unmapped_records:
        unmapped_csv_path = os.path.join(output_dir, "unmapped_files.csv")
        with open(unmapped_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(unmapped_records[0].keys()))
            writer.writeheader()
            writer.writerows(unmapped_records)

    if excluded_records:
        excluded_csv_path = os.path.join(output_dir, "excluded_files.csv")
        with open(excluded_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(excluded_records[0].keys()))
            writer.writeheader()
            writer.writerows(excluded_records)

    return audit_summary
