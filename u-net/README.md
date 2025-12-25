# ACL tear Segmentation with U-Net

This repository contains code and dataset organization for **automatic segmentation of the anterior cruciate ligament (ACL) in 2D knee MRI scans** using a U-Net architecture.

## Repository Structure

```
unet/
├── dataset.py                # Dataset class for loading images and masks
├── model.py                  # U-Net model definition
├── helpers.py                # Utility functions
└── main.py                   # Training and evaluation pipeline
```

---

## Dataset

- **kneeMRI-ACL/**: Contains 120 patient folders (0–119). Each folder includes:

  - `img.png`: 2D MRI slice
  - `msk.png`: Corresponding segmentation mask of ACL

- **train.txt / val.txt / test.txt**: Text files listing indices of training, validation, and test samples.

---

## Requirements

You can install dependencies via:

```bash
pip install -r requirements.txt
```

---

## Usage

1. **Prepare dataset:** Place the `dataset/kneeMRI-ACL` folder and txt files in the `dataset/` directory.
2. **Configure parameters:** Edit `main.py` for training hyperparameters, paths, and device selection.
3. **Run training:**

```bash
python main.py
```

---

## Citation / Acknowledgements

If you use this dataset or baseline code in your research, please cite our dataset paper:

TBD
