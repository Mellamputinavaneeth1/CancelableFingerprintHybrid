import os
import re
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

SOCOFING_DPI = 500.0

def parse_socofing_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Parses standard SOCOFing filename:
    Real:    {id}__{gender}_{hand}_{finger_name}_finger.BMP
             e.g., '1__M_Left_index_finger.BMP'
    Altered: {id}__{gender}_{hand}_{finger_name}_finger_{alteration}.BMP
             e.g., '1__M_Left_index_finger_Obl.BMP' or '1__M_Left_index_finger_CR.BMP'
    """
    basename = os.path.splitext(filename)[0]
    # Check if altered
    pattern_altered = r"^(\d+)__([MF])_([A-Za-z]+)_([A-Za-z]+)_finger_([A-Za-z0-9_]+)$"
    m_alt = re.match(pattern_altered, basename)
    if m_alt:
        alt_code = m_alt.group(5)
        # Classify alteration class
        if "Obl" in alt_code or "obliteration" in alt_code.lower():
            alt_class = "Obliteration"
        elif "CR" in alt_code or "rotation" in alt_code.lower():
            alt_class = "Central_Rotation"
        elif "Zcut" in alt_code or "zcut" in alt_code.lower() or "z_cut" in alt_code.lower():
            alt_class = "Z-cut"
        else:
            alt_class = alt_code
            
        return {
            "subject_id": int(m_alt.group(1)),
            "gender": m_alt.group(2),
            "hand": m_alt.group(3),
            "finger": m_alt.group(4),
            "is_altered": True,
            "alteration_code": alt_code,
            "alteration_class": alt_class
        }
        
    pattern_real = r"^(\d+)__([MF])_([A-Za-z]+)_([A-Za-z]+)_finger$"
    m_real = re.match(pattern_real, basename)
    if m_real:
        return {
            "subject_id": int(m_real.group(1)),
            "gender": m_real.group(2),
            "hand": m_real.group(3),
            "finger": m_real.group(4),
            "is_altered": False,
            "alteration_code": "None",
            "alteration_class": "None"
        }
        
    return None


def load_socofing_dataset(root_dir: str = "dataset/SOCOFing") -> Dict[str, Any]:
    """
    Loads and indexes the SOCOFing dataset from disk.
    Supports either root/Real and root/Altered subfolders, or flat directories.
    """
    if not os.path.exists(root_dir):
        return {
            "status": "not_found",
            "root_dir": root_dir,
            "real_images": [],
            "altered_images": [],
            "subjects": {}
        }
        
    real_images: List[Dict[str, Any]] = []
    altered_images: List[Dict[str, Any]] = []
    subjects: Dict[int, Dict[str, Any]] = {}
    
    # Walk directory to find all BMP / PNG / TIF images
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith(('.bmp', '.png', '.tif', '.tiff')):
                continue
            meta = parse_socofing_filename(fname)
            if not meta:
                continue
                
            fpath = os.path.join(dirpath, fname)
            meta["path"] = fpath
            meta["filename"] = fname
            meta["dpi"] = SOCOFING_DPI
            
            # Determine difficulty if in an Altered-Easy/Medium/Hard path
            lower_path = dirpath.lower()
            if "easy" in lower_path:
                meta["difficulty"] = "Easy"
            elif "medium" in lower_path:
                meta["difficulty"] = "Medium"
            elif "hard" in lower_path:
                meta["difficulty"] = "Hard"
            else:
                meta["difficulty"] = "Unknown"
                
            subj_id = meta["subject_id"]
            if subj_id not in subjects:
                subjects[subj_id] = {"real": [], "altered": []}
                
            if meta["is_altered"]:
                altered_images.append(meta)
                subjects[subj_id]["altered"].append(meta)
            else:
                real_images.append(meta)
                subjects[subj_id]["real"].append(meta)
                
    return {
        "status": "loaded" if real_images else "empty",
        "root_dir": root_dir,
        "dpi": SOCOFING_DPI,
        "real_images": real_images,
        "altered_images": altered_images,
        "subjects": subjects,
        "num_subjects": len(subjects),
        "total_real": len(real_images),
        "total_altered": len(altered_images)
    }


def apply_synthetic_obliteration(image: np.ndarray, difficulty: str = "Medium", rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Applies realistic elliptical obliteration scar/abrasion with edge feathering."""
    if rng is None:
        rng = np.random.RandomState(42)
    h, w = image.shape
    out = image.copy()
    
    num_scars = {"Easy": 1, "Medium": 2, "Hard": 3}.get(difficulty, 2)
    frac = {"Easy": 0.15, "Medium": 0.25, "Hard": 0.35}.get(difficulty, 0.25)
    
    for _ in range(num_scars):
        cx = rng.randint(int(w * 0.25), int(w * 0.75))
        cy = rng.randint(int(h * 0.25), int(h * 0.75))
        axes = (int(w * frac * rng.uniform(0.6, 1.0)), int(h * frac * rng.uniform(0.6, 1.0)))
        angle = rng.uniform(0, 180)
        
        # Create elliptical scar mask
        scar_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(scar_mask, (cx, cy), axes, angle, 0, 360, 255, -1)
        
        # Obliterate with textured white/gray noise simulating scar tissue
        noise = rng.normal(210, 25, (h, w)).clip(0, 255).astype(np.uint8)
        out[scar_mask > 0] = noise[scar_mask > 0]
        
    return out


def apply_synthetic_central_rotation(image: np.ndarray, difficulty: str = "Medium", rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Applies realistic central core skin twist rotation."""
    if rng is None:
        rng = np.random.RandomState(42)
    h, w = image.shape
    out = image.copy()
    
    rot_angle = {"Easy": 15.0, "Medium": 30.0, "Hard": 45.0}.get(difficulty, 30.0)
    if rng.rand() > 0.5:
        rot_angle = -rot_angle
        
    cx, cy = w // 2 + rng.randint(-15, 15), h // 2 + rng.randint(-15, 15)
    radius = int(min(h, w) * {"Easy": 0.25, "Medium": 0.35, "Hard": 0.45}.get(difficulty, 0.35))
    
    # Create circular core mask
    core_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(core_mask, (cx, cy), radius, 255, -1)
    
    # Rotate full image around core center
    M = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Seamless blend / feather edges of rotated core
    dist = cv2.distanceTransform(core_mask, cv2.DIST_L2, 5)
    max_d = dist.max()
    if max_d > 0:
        alpha = np.clip(dist / (max_d * 0.3), 0.0, 1.0)
        out = (alpha * rotated + (1.0 - alpha) * image).astype(np.uint8)
    else:
        out[core_mask > 0] = rotated[core_mask > 0]
        
    return out


def apply_synthetic_zcut(image: np.ndarray, difficulty: str = "Medium", rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Applies surgical Z-cut distortion: shears skin along a Z-shaped incision seam."""
    if rng is None:
        rng = np.random.RandomState(42)
    h, w = image.shape
    out = image.copy()
    
    shear_shift = {"Easy": 8, "Medium": 16, "Hard": 24}.get(difficulty, 16)
    y_top = int(h * 0.3)
    y_mid = int(h * 0.5)
    y_bot = int(h * 0.7)
    
    # Upper flap shifted right
    out[:y_top, :] = np.roll(image[:y_top, :], shear_shift, axis=1)
    # Middle diagonal flap sheared
    out[y_top:y_bot, :] = np.roll(image[y_top:y_bot, :], -shear_shift, axis=1)
    
    # Add scar incision seam along boundary
    cv2.line(out, (0, y_top), (w, y_top), 240, 2)
    cv2.line(out, (w, y_top), (0, y_bot), 240, 2)
    cv2.line(out, (0, y_bot), (w, y_bot), 240, 2)
    
    return out


def generate_synthetic_socofing_benchmark(target_dir: str = "dataset/SOCOFing", 
                                          source_images: Optional[List[Tuple[str, np.ndarray]]] = None,
                                          num_subjects: int = 10,
                                          seed: int = 42) -> Dict[str, Any]:
    """
    Synthesizes a standardized SOCOFing benchmark hierarchy from base fingerprint images.
    Creates Real and Altered-{Easy, Medium, Hard} subsets under target_dir.
    """
    rng = np.random.RandomState(seed)
    real_dir = os.path.join(target_dir, "Real")
    altered_base = os.path.join(target_dir, "Altered")
    
    os.makedirs(real_dir, exist_ok=True)
    for diff in ["Easy", "Medium", "Hard"]:
        os.makedirs(os.path.join(altered_base, f"Altered-{diff}"), exist_ok=True)
        
    if not source_images:
        # Generate procedural base fingerprints if no source images passed
        source_images = []
        for i in range(num_subjects):
            h, w = 388, 374
            y, x = np.mgrid[0:h, 0:w]
            cx, cy = w // 2, h // 2
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            theta = np.arctan2(y - cy, x - cx)
            freq = 0.08 + (i % 3) * 0.01
            pattern = 128 + 100 * np.sin(freq * r + 2 * theta + rng.normal(0, 0.1, (h, w)))
            pattern = pattern.clip(0, 255).astype(np.uint8)
            source_images.append((f"base_{i+1}", pattern))
            
    created_real = 0
    created_altered = 0
    
    for idx, (name, img) in enumerate(source_images[:num_subjects]):
        subj_id = idx + 1
        gender = "M" if subj_id % 2 == 1 else "F"
        hand = "Left" if (subj_id // 5) % 2 == 0 else "Right"
        finger_names = ["thumb", "index", "middle", "ring", "little"]
        finger = finger_names[subj_id % 5]
        
        real_fname = f"{subj_id}__{gender}_{hand}_{finger}_finger.BMP"
        real_path = os.path.join(real_dir, real_fname)
        cv2.imwrite(real_path, img)
        created_real += 1
        
        for diff in ["Easy", "Medium", "Hard"]:
            diff_dir = os.path.join(altered_base, f"Altered-{diff}")
            
            # 1. Obliteration
            img_obl = apply_synthetic_obliteration(img, difficulty=diff, rng=np.random.RandomState(subj_id * 10 + 1))
            fname_obl = f"{subj_id}__{gender}_{hand}_{finger}_finger_Obl.BMP"
            cv2.imwrite(os.path.join(diff_dir, fname_obl), img_obl)
            
            # 2. Central Rotation
            img_cr = apply_synthetic_central_rotation(img, difficulty=diff, rng=np.random.RandomState(subj_id * 10 + 2))
            fname_cr = f"{subj_id}__{gender}_{hand}_{finger}_finger_CR.BMP"
            cv2.imwrite(os.path.join(diff_dir, fname_cr), img_cr)
            
            # 3. Z-cut
            img_zcut = apply_synthetic_zcut(img, difficulty=diff, rng=np.random.RandomState(subj_id * 10 + 3))
            fname_zcut = f"{subj_id}__{gender}_{hand}_{finger}_finger_Zcut.BMP"
            cv2.imwrite(os.path.join(diff_dir, fname_zcut), img_zcut)
            
            created_altered += 3
            
    return {
        "status": "generated",
        "target_dir": target_dir,
        "num_subjects": min(num_subjects, len(source_images)),
        "created_real": created_real,
        "created_altered": created_altered
    }
