import os
import csv
from typing import List, Tuple, Dict
import numpy as np

def generate_pair_lists(finger_ids: List[str], impressions_per_finger: int, 
                        output_dir: str = "experiments/pair_lists") -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Generates reproducible genuine and impostor pair lists for FVC/SOCOFing evaluation.
    
    Args:
        finger_ids: List of unique finger identifiers (e.g., ['101_1', '101_2', ...]).
        impressions_per_finger: Number of impressions per finger (e.g., 8 for FVC).
        output_dir: Directory to store genuine_pairs.csv and impostor_pairs.csv.
        
    Returns:
        genuine_pairs: List of (img_path_a, img_path_b) tuples.
        impostor_pairs: List of (img_path_a, img_path_b) tuples.
    """
    os.makedirs(output_dir, exist_ok=True)
    genuine_pairs = []
    impostor_pairs = []
    
    # Genuine pairs: All unique combinations of impressions for each finger
    for fid in finger_ids:
        for imp1 in range(1, impressions_per_finger + 1):
            for imp2 in range(imp1 + 1, impressions_per_finger + 1):
                img1 = f"{fid}_{imp1}.tif"
                img2 = f"{fid}_{imp2}.tif"
                genuine_pairs.append((img1, img2))
                
    # Impostor pairs: First impression of each finger compared against first impression of all other fingers
    n_fingers = len(finger_ids)
    for i in range(n_fingers):
        for j in range(i + 1, n_fingers):
            img1 = f"{finger_ids[i]}_1.tif"
            img2 = f"{finger_ids[j]}_1.tif"
            impostor_pairs.append((img1, img2))
            
    # Save to CSV
    gen_path = os.path.join(output_dir, "genuine_pairs.csv")
    imp_path = os.path.join(output_dir, "impostor_pairs.csv")
    
    with open(gen_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_1", "image_2", "is_genuine"])
        for p in genuine_pairs:
            writer.writerow([p[0], p[1], 1])
            
    with open(imp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_1", "image_2", "is_genuine"])
        for p in impostor_pairs:
            writer.writerow([p[0], p[1], 0])
            
    return genuine_pairs, impostor_pairs

def load_pair_lists(output_dir: str = "experiments/pair_lists") -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Loads frozen genuine and impostor pair lists from disk."""
    gen_path = os.path.join(output_dir, "genuine_pairs.csv")
    imp_path = os.path.join(output_dir, "impostor_pairs.csv")
    
    genuine_pairs = []
    impostor_pairs = []
    
    if os.path.exists(gen_path):
        with open(gen_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                genuine_pairs.append((row[0], row[1]))
                
    if os.path.exists(imp_path):
        with open(imp_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                impostor_pairs.append((row[0], row[1]))
                
    return genuine_pairs, impostor_pairs
