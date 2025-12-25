from ultralytics import YOLO, settings
import mlflow
import yaml
from mlflow.tracking import MlflowClient

with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)

settings.update({"mlflow": True})
mlflow.set_tracking_uri("http://localhost:5555")
mlflow.set_experiment("knee_ultralytics_segmentation")
run_name = f"opt_{config['optimizer']}_lr_{config['learning_rate']}"

with mlflow.start_run(run_name=run_name):
    pretrained = './pretrained_weights/yolo11x-seg.pt'
    model = YOLO(pretrained)
    model.train(
        data=config['data_path'],
        epochs=config['epochs'],
        imgsz=config['img_size'],
        patience=config['patience'],
        device='0',
        optimizer=config['optimizer'],
        lr0=config['learning_rate'],
        weight_decay=config['weight_decay'],
        verbose=True,
        save=True
    )
