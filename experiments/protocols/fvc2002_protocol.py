"""
FVC2002 standard matching protocol generator.
Implements the official FVC comparison rules:
- Genuine: all pairs of impressions (imp_i, imp_j with i < j) for each finger (28 pairs/finger).
- Impostor: first impression of finger u vs first impression of finger v (u < v).
"""

from typing import Dict, Any, List, Tuple
import os

def generate_fvc_protocol_pairs(fvc_data: Dict[str, Any], 
                                all_impostor_impressions: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generates genuine and impostor comparison pairs according to the official FVC protocol.
    
    Returns:
        (genuine_pairs, impostor_pairs)
    """
    fingers = fvc_data["fingers"]
    subset = fvc_data["subset"]
    finger_ids = sorted(list(fingers.keys()))
    
    genuine_pairs: List[Dict[str, Any]] = []
    impostor_pairs: List[Dict[str, Any]] = []
    
    pair_counter = 0
    
    # 1. Genuine pairs: all (i, j) combinations with i < j for each finger
    for fid in finger_ids:
        imps = sorted(list(fingers[fid].keys()))
        n_imps = len(imps)
        for i in range(n_imps):
            for j in range(i + 1, n_imps):
                imp_i = imps[i]
                imp_j = imps[j]
                genuine_pairs.append({
                    "pair_id": pair_counter,
                    "genuine": 1,
                    "subset": subset,
                    "probe_finger_id": fid,
                    "probe_impression_id": imp_j,
                    "probe_path": fingers[fid][imp_j],
                    "gallery_finger_id": fid,
                    "gallery_impression_id": imp_i,
                    "gallery_path": fingers[fid][imp_i]
                })
                pair_counter += 1
                
    # 2. Impostor pairs: official FVC protocol uses 1st impression across distinct fingers
    n_fingers = len(finger_ids)
    if not all_impostor_impressions:
        # Standard FVC: imp 1 of finger u vs imp 1 of finger v (u < v)
        for i in range(n_fingers):
            fid_u = finger_ids[i]
            if 1 not in fingers[fid_u]:
                continue
            path_u = fingers[fid_u][1]
            for j in range(i + 1, n_fingers):
                fid_v = finger_ids[j]
                if 1 not in fingers[fid_v]:
                    continue
                path_v = fingers[fid_v][1]
                impostor_pairs.append({
                    "pair_id": pair_counter,
                    "genuine": 0,
                    "subset": subset,
                    "probe_finger_id": fid_v,
                    "probe_impression_id": 1,
                    "probe_path": path_v,
                    "gallery_finger_id": fid_u,
                    "gallery_impression_id": 1,
                    "gallery_path": path_u
                })
                pair_counter += 1
    else:
        # Exhaustive impostor pairs across all impressions (for dense analysis)
        for i in range(n_fingers):
            fid_u = finger_ids[i]
            for j in range(i + 1, n_fingers):
                fid_v = finger_ids[j]
                for imp_u, path_u in fingers[fid_u].items():
                    for imp_v, path_v in fingers[fid_v].items():
                        impostor_pairs.append({
                            "pair_id": pair_counter,
                            "genuine": 0,
                            "subset": subset,
                            "probe_finger_id": fid_v,
                            "probe_impression_id": imp_v,
                            "probe_path": path_v,
                            "gallery_finger_id": fid_u,
                            "gallery_impression_id": imp_u,
                            "gallery_path": path_u
                        })
                        pair_counter += 1
                        
    return genuine_pairs, impostor_pairs
