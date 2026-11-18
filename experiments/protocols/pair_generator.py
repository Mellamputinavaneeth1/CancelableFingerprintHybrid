import os
import csv
from typing import List, Dict, Any, Tuple

def export_manifest_csv(pairs: List[Dict[str, Any]], output_path: str):
    """
    Exports a list of comparison pairs to a frozen CSV manifest.
    """
    if not pairs:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    keys = list(pairs[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(pairs)

def load_manifest_csv(input_path: str) -> List[Dict[str, Any]]:
    """
    Loads comparison pairs from a CSV manifest.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Manifest not found: {input_path}")
    pairs = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row = dict(row)
            parsed_row["pair_id"] = int(parsed_row["pair_id"])
            parsed_row["genuine"] = int(parsed_row["genuine"])
            if "probe_finger_id" in parsed_row:
                parsed_row["probe_finger_id"] = int(parsed_row["probe_finger_id"])
                parsed_row["gallery_finger_id"] = int(parsed_row["gallery_finger_id"])
                parsed_row["probe_impression_id"] = int(parsed_row["probe_impression_id"])
                parsed_row["gallery_impression_id"] = int(parsed_row["gallery_impression_id"])
            pairs.append(parsed_row)
    return pairs
