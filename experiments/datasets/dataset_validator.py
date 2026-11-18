import os
from typing import Dict, Any, List, Set, Tuple
import cv2
import numpy as np

def validate_fvc_dataset(fvc_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates structural, dimensional, and pixel integrity of an FVC dataset.
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    subset = fvc_data.get("subset", "Unknown")
    dpi = fvc_data.get("dpi", 500.0)
    expected_shape = fvc_data.get("expected_shape", None)
    all_images = fvc_data.get("all_images", [])
    fingers = fvc_data.get("fingers", {})
    
    if not all_images:
        errors.append(f"Dataset {subset} has 0 images.")
        return {"valid": False, "errors": errors, "warnings": warnings}
        
    is_set_b = "B" in subset.upper()
    expected_fingers = 10 if is_set_b else 100
    expected_impressions = 8
    expected_total = expected_fingers * expected_impressions
    
    if len(fingers) != expected_fingers:
        warnings.append(f"Expected {expected_fingers} fingers, found {len(fingers)}.")
        
    if len(all_images) != expected_total:
        warnings.append(f"Expected {expected_total} images, found {len(all_images)}.")
        
    # Validate each image
    verified_images = 0
    corrupted_images = 0
    
    for item in all_images:
        fpath = item["path"]
        if not os.path.exists(fpath):
            errors.append(f"File missing: {fpath}")
            continue
            
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            errors.append(f"Corrupted image, failed to read: {fpath}")
            corrupted_images += 1
            continue
            
        h, w = img.shape
        if expected_shape and (h, w) != expected_shape:
            # Check transposed dimensions
            if (w, h) == expected_shape:
                warnings.append(f"Image {item['filename']} transposed: got ({h}, {w}), expected {expected_shape}.")
            else:
                warnings.append(f"Image {item['filename']} dimension mismatch: got ({h}, {w}), expected {expected_shape}.")
                
        # Check standard deviation (detect blank or solid-color corruptions)
        std = float(np.std(img))
        if std < 5.0:
            errors.append(f"Image {item['filename']} has near-zero variance (std={std:.2f}): possibly blank or corrupted.")
            
        verified_images += 1
        
    # Check completeness per finger
    incomplete_fingers = []
    for f_id, imps in fingers.items():
        if len(imps) != expected_impressions:
            incomplete_fingers.append((f_id, len(imps)))
            
    if incomplete_fingers:
        warnings.append(f"Incomplete impression sets for fingers: {incomplete_fingers[:5]}...")
        
    is_valid = len(errors) == 0
    return {
        "valid": is_valid,
        "subset": subset,
        "dpi": dpi,
        "num_fingers": len(fingers),
        "total_images": len(all_images),
        "verified_images": verified_images,
        "corrupted_images": corrupted_images,
        "errors": errors,
        "warnings": warnings
    }


def validate_socofing_dataset(socofing_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates SOCOFing real and altered image pairs, metadata, and pixel integrity.
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    real_images = socofing_data.get("real_images", [])
    altered_images = socofing_data.get("altered_images", [])
    subjects = socofing_data.get("subjects", {})
    
    if not real_images:
        errors.append("No real gallery images found in SOCOFing.")
        return {"valid": False, "errors": errors, "warnings": warnings}
        
    # Check each real image
    for r in real_images:
        fpath = r["path"]
        if not os.path.exists(fpath):
            errors.append(f"Real image missing: {fpath}")
            continue
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            errors.append(f"Corrupted real image: {fpath}")
            
    # Check each altered image has a corresponding subject
    orphan_altered = 0
    for a in altered_images:
        sid = a["subject_id"]
        if sid not in subjects or not subjects[sid]["real"]:
            orphan_altered += 1
            
    if orphan_altered > 0:
        warnings.append(f"{orphan_altered} altered images have no matching enrolled real gallery image.")
        
    is_valid = len(errors) == 0
    return {
        "valid": is_valid,
        "num_subjects": len(subjects),
        "total_real": len(real_images),
        "total_altered": len(altered_images),
        "orphan_altered": orphan_altered,
        "errors": errors,
        "warnings": warnings
    }


def validate_split_disjointness(train_ids: List[int], val_ids: List[int], test_ids: List[int]) -> bool:
    """
    Verifies that subject IDs across train, val, and test splits are strictly disjoint.
    """
    s_train = set(train_ids)
    s_val = set(val_ids)
    s_test = set(test_ids)
    
    overlap_train_val = s_train.intersection(s_val)
    overlap_train_test = s_train.intersection(s_test)
    overlap_val_test = s_val.intersection(s_test)
    
    return len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0
