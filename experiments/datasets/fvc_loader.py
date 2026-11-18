import os
import zipfile
import urllib.request
import re
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

FVC_SUBSETS = {
    "DB1_A": {
        "dpi": 500.0,
        "width": 388,
        "height": 374,
        "sensor": "Optical (TouchView II by Identix)",
        "url": "https://ndownloader.figshare.com/files/44522927",
        "is_set_b": False
    },
    "DB1_B": {
        "dpi": 500.0,
        "width": 388,
        "height": 374,
        "sensor": "Optical (TouchView II by Identix)",
        "url": "http://bias.csr.unibo.it/fvc2002/Downloads/DB1_B.zip",
        "is_set_b": True
    },
    "DB2_A": {
        "dpi": 569.0,
        "width": 296,
        "height": 560,
        "sensor": "Optical (FX2000 by Biometrika)",
        "url": None,
        "is_set_b": False
    },
    "DB2_B": {
        "dpi": 569.0,
        "width": 296,
        "height": 560,
        "sensor": "Optical (FX2000 by Biometrika)",
        "url": "http://bias.csr.unibo.it/fvc2002/Downloads/DB2_B.zip",
        "is_set_b": True
    },
    "DB3_A": {
        "dpi": 500.0,
        "width": 300,
        "height": 300,
        "sensor": "Capacitive (100 SC by Precise Biometrics)",
        "url": None,
        "is_set_b": False
    },
    "DB3_B": {
        "dpi": 500.0,
        "width": 300,
        "height": 300,
        "sensor": "Capacitive (100 SC by Precise Biometrics)",
        "url": "http://bias.csr.unibo.it/fvc2002/Downloads/DB3_B.zip",
        "is_set_b": True
    },
    "DB4_A": {
        "dpi": 500.0,
        "width": 288,
        "height": 384,
        "sensor": "Synthetic Generator (SFinGe v2.51)",
        "url": None,
        "is_set_b": False
    },
    "DB4_B": {
        "dpi": 500.0,
        "width": 288,
        "height": 384,
        "sensor": "Synthetic Generator (SFinGe v2.51)",
        "url": "http://bias.csr.unibo.it/fvc2002/Downloads/DB4_B.zip",
        "is_set_b": True
    }
}


def download_fvc_subset(subset: str, root_dir: str = "dataset/FVC2002", force: bool = False) -> str:
    """
    Downloads and extracts an FVC2002 subset if not already present.
    Returns the path to the subset directory containing .tif files.
    """
    if subset not in FVC_SUBSETS:
        raise ValueError(f"Unknown FVC2002 subset: {subset}. Expected one of {list(FVC_SUBSETS.keys())}")
        
    subset_dir = os.path.join(root_dir, subset)
    os.makedirs(subset_dir, exist_ok=True)
    
    # Check if images already exist
    existing_images = [f for f in os.listdir(subset_dir) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png'))]
    if existing_images and not force:
        return subset_dir
        
    url = FVC_SUBSETS[subset]["url"]
    if not url:
        raise RuntimeError(
            f"Automated download URL not available for {subset} (Set A requires license/manual download). "
            f"Please place .tif files in '{subset_dir}'."
        )
        
    zip_path = os.path.join(root_dir, f"{subset}.zip")
    print(f"Downloading {subset} from {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req, timeout=60) as response, open(zip_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
        
    print(f"Extracting {zip_path} into {subset_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            filename = os.path.basename(member)
            if not filename:
                continue
            # Extract directly into subset_dir flattening any inner directory structure
            source = zip_ref.open(member)
            target = open(os.path.join(subset_dir, filename), "wb")
            with source, target:
                target.write(source.read())
                
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    return subset_dir


def parse_fvc_filename(filename: str) -> Optional[Tuple[int, int]]:
    """
    Parses FVC filename like '101_1.tif' or '1_8.tif' into (finger_id, impression_id).
    Returns None if pattern does not match.
    """
    basename = os.path.splitext(filename)[0]
    match = re.match(r"^(\d+)_(\d+)$", basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def load_fvc_dataset(subset: str, root_dir: str = "dataset/FVC2002", auto_download: bool = True) -> Dict[str, Any]:
    """
    Loads and structures an FVC2002 dataset.
    Returns:
        {
            "subset": str,
            "dpi": float,
            "sensor": str,
            "expected_shape": (height, width),
            "subset_dir": str,
            "fingers": Dict[int, Dict[int, str]],  # finger_id -> {impression_id: file_path}
            "all_images": List[Dict[str, Any]],
            "num_fingers": int,
            "total_images": int
        }
    """
    if subset not in FVC_SUBSETS:
        raise ValueError(f"Unknown FVC2002 subset: {subset}. Expected one of {list(FVC_SUBSETS.keys())}")
        
    meta = FVC_SUBSETS[subset]
    subset_dir = os.path.join(root_dir, subset)
    
    if auto_download and (not os.path.exists(subset_dir) or not [f for f in os.listdir(subset_dir) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png'))]):
        download_fvc_subset(subset, root_dir=root_dir)
        
    if not os.path.exists(subset_dir):
        raise FileNotFoundError(f"Subset directory not found: {subset_dir}")
        
    fingers: Dict[int, Dict[int, str]] = {}
    all_images: List[Dict[str, Any]] = []
    
    for fname in sorted(os.listdir(subset_dir)):
        if not fname.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
            continue
        parsed = parse_fvc_filename(fname)
        if not parsed:
            continue
        finger_id, imp_id = parsed
        fpath = os.path.join(subset_dir, fname)
        
        if finger_id not in fingers:
            fingers[finger_id] = {}
        fingers[finger_id][imp_id] = fpath
        
        all_images.append({
            "finger_id": finger_id,
            "impression_id": imp_id,
            "path": fpath,
            "filename": fname,
            "subset": subset,
            "dpi": meta["dpi"]
        })
        
    return {
        "subset": subset,
        "dpi": meta["dpi"],
        "sensor": meta["sensor"],
        "expected_shape": (meta["height"], meta["width"]),
        "subset_dir": subset_dir,
        "fingers": fingers,
        "all_images": all_images,
        "num_fingers": len(fingers),
        "total_images": len(all_images)
    }


def read_fvc_image(path: str) -> np.ndarray:
    """Reads a grayscale fingerprint image from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to decode image at {path}")
    return img
