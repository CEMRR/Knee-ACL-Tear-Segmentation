# ACL tear Segmentation with YOLO11 (MLflow Tracking)

This project implements a YOLO11 segmentation training pipeline using Ultralytics,
with experiment tracking via MLflow and configuration through a YAML file.

---

## 📁 Project Structure

```
├── config.yaml                  # YAML file with training configuration
├── requirements.txt             # List of Python dependencies
├── train.py                     # Training script using Ultralytics and MLflow
├── yolo_dataset/
│   └── dataset.yaml             # Dataset paths and class information for YOLO
```

---

## 🔧 Configuration (`config.yaml`)

```yaml
data_path: './yolo_dataset/dataset.yaml'
learning_rate: 0.01
patience: 100
epochs: 500
img_size: 512
weight_decay: 0
optimizer: sgd
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start MLflow Tracking Server (Optional)

```bash
mlflow server --backend-store-uri /mlflow --host 0.0.0.0 --port 5555
```

Visit MLflow dashboard at: [http://localhost:5555](http://localhost:5555)

### 3. Run Training

```bash
python train.py
```

Training parameters will be loaded from `config.yaml`, and all metrics will be logged to MLflow.

---

## 🧪 Evaluation

Evaluate the trained model using:

```bash
python test.py
```

---

## 📌 Notes

- Use `prepare_db.py` to prepare dataset for training the model
- YOLO11 mask format is used in `labels/*.txt`.
- Ensure `dataset.yaml` includes:
  - `train`: path to training images
  - `val`: path to validation images
  - `test`: path to test images
  - `names`: class names

---

## Citation / Acknowledgements

If you use this dataset or baseline code in your research, please cite our dataset paper:

TBD
