[![DOI](https://zenodo.org/badge/1027561924.svg)](https://doi.org/10.5281/zenodo.18054534)

# Knee MRI ACL Dataset and Baseline Code

This repository accompanies our **Dataset for Anterior Cruciate Ligament Tears Segmentation in Knee MRI** paper and provides:

1. **The curated ACL MRI dataset**
2. **Baseline code** for ACL tear segmentation in 2D MRI slices:
   - **U-Net segmentation**
   - **YOLO segmentation**

---

## Dataset Overview

- **kneeMRI-ACL/**: Contains 110 patient folders (0–109). Each folder includes:
  - `img.png`: 2D MRI slice
  - `msk.png`: corresponding ground-truth mask
- **train.txt / val.txt / test.txt**: text files containing indices of the samples in each split

```
dataset/
├── kneeMRI-ACL/ # MRI dataset
│ ├── 0/
│ │ ├── img.png # input MRI image
│ │ └── msk.png # corresponding ground-truth mask
│ ├── 1/
│ └── ... # up to 109
├── train.txt # list of training sample indices
├── val.txt # list of validation sample indices
└── test.txt # list of test sample indices
```

---

## Baseline Code

### U-Net Segmentation (`unet/`)

- Implements a **baseline U-Net** for ACL tear segmentation in 2D MRI slices
- For detailed usage, see `unet/README.md`

### YOLO Segmentation (`yolo/`)

- Implements a **baseline YOLO** for ACL tear segmentation in 2D MRI slices
- Dataset organization and usage instructions will be provided in `yolo/README.md`

---

## Citation / Acknowledgements

If you use this dataset or baseline code in your research, please cite our dataset paper:

TBD
