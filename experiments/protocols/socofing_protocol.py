"""
SOCOFing matching protocol generator.
Pairs enrolled real gallery prints against altered probe prints
across Easy, Medium, and Hard alterations (Obliteration, Central Rotation, Z-cut).
"""

from typing import Dict, Any, List, Tuple
import numpy as np

def generate_socofing_protocol_pairs(socofing_data: Dict[str, Any],
                                     max_impostors_per_probe: int = 5,
                                     seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generates genuine and impostor comparison pairs for SOCOFing.
    
    Genuine: Probe altered image vs Gallery real image of same subject & finger.
    Impostor: Probe altered image vs Gallery real image of different subjects.
    
    Returns:
        (genuine_pairs, impostor_pairs)
    """
    rng = np.random.RandomState(seed)
    subjects = socofing_data.get("subjects", {})
    all_subject_ids = sorted(list(subjects.keys()))
    
    genuine_pairs: List[Dict[str, Any]] = []
    impostor_pairs: List[Dict[str, Any]] = []
    pair_counter = 0
    
    for subj_id in all_subject_ids:
        reals = subjects[subj_id]["real"]
        altereds = subjects[subj_id]["altered"]
        if not reals or not altereds:
            continue
            
        gallery_item = reals[0]
        
        # Genuine comparisons
        for alt_item in altereds:
            genuine_pairs.append({
                "pair_id": pair_counter,
                "genuine": 1,
                "subject_id": subj_id,
                "probe_path": alt_item["path"],
                "probe_filename": alt_item["filename"],
                "difficulty": alt_item.get("difficulty", "Unknown"),
                "alteration_class": alt_item.get("alteration_class", "Unknown"),
                "gallery_path": gallery_item["path"],
                "gallery_filename": gallery_item["filename"],
                "gallery_subject_id": subj_id
            })
            pair_counter += 1
            
            # Impostor comparisons: pick random other subjects' gallery images
            other_subjs = [s for s in all_subject_ids if s != subj_id and subjects[s]["real"]]
            if other_subjs:
                chosen_impostors = rng.choice(other_subjs, size=min(max_impostors_per_probe, len(other_subjs)), replace=False)
                for imp_subj in chosen_impostors:
                    imp_gallery = subjects[imp_subj]["real"][0]
                    impostor_pairs.append({
                        "pair_id": pair_counter,
                        "genuine": 0,
                        "subject_id": subj_id,
                        "probe_path": alt_item["path"],
                        "probe_filename": alt_item["filename"],
                        "difficulty": alt_item.get("difficulty", "Unknown"),
                        "alteration_class": alt_item.get("alteration_class", "Unknown"),
                        "gallery_path": imp_gallery["path"],
                        "gallery_filename": imp_gallery["filename"],
                        "gallery_subject_id": imp_subj
                    })
                    pair_counter += 1
                    
    return genuine_pairs, impostor_pairs
