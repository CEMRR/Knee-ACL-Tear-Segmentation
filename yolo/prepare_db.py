import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import Optional

# =========================
# CONFIG
# =========================
RAW_DATA_DIR = Path("dataset/kneeMRI-ACL")      
SPLITS_DIR = Path("dataset")             
OUTPUT_DIR = Path("yolo_dataset")
CLASS_ID = 0                        

# =========================
# CREATE YOLO STRUCTURE
# =========================
for split in ["train", "val", "test"]:
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# =========================
# MASK → YOLO SEGMENTATION
# =========================
def mask_to_yolo_segmentation(mask_path: Path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")

    h, w = mask.shape
    _, bin_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    yolo_lines = []

    for cnt in contours:
        if len(cnt) < 3:
            continue

        cnt = cnt.squeeze(1)  # (N,2)

        normalized = []
        for x, y in cnt:
            normalized.append(f"{x / w:.6f} {y / h:.6f}")

        line = f"{CLASS_ID} " + " ".join(normalized)
        yolo_lines.append(line)

    return yolo_lines

# =========================
# PROCESS SPLITS
# =========================
for split in ["train", "val", "test"]:
    split_file = SPLITS_DIR / f"{split}.txt"
    if not split_file.exists():
        print(f"Missing {split_file}, skipping")
        continue

    with open(split_file) as f:
        ids = [line.strip() for line in f if line.strip()]

    for sample_id in ids:
        src_dir = RAW_DATA_DIR / sample_id
        img_src = src_dir / "img.png"
        mask_src = src_dir / "mask.png"

        if not img_src.exists() or not mask_src.exists():
            print(f"Missing data for ID {sample_id}, skipping")
            continue

        # ---------- IMAGE ----------
        img_name = f"img_{sample_id}.png"
        img_dst = OUTPUT_DIR / "images" / split / img_name
        shutil.copy(img_src, img_dst)

        # ---------- LABEL ----------
        yolo_lines = mask_to_yolo_segmentation(mask_src)
        label_dst = OUTPUT_DIR / "labels" / split / f"img_{sample_id}.txt"

        with open(label_dst, "w") as f:
            f.write("\n".join(yolo_lines))

print("YOLO segmentation dataset created successfully")
