from ultralytics import YOLO
import yaml

with open('./config.yaml', 'rb') as f:
    config = yaml.safe_load(f)

model = YOLO("./pretrained_weights/best.pt")
metrics = model.val(data="./yolo_dataset/dataset.yaml", split="test", imgsz=config['img_size'], save=True, save_txt=True, save_json=True)

results_dict = metrics.results_dict

precision = results_dict['metrics/precision(M)']
recall = results_dict['metrics/recall(M)']

f1_score   = 2 * (precision * recall) / (precision + recall + 1e-8)
dice_score = f1_score
iou_score  = (precision * recall) / (precision + recall - precision * recall + 1e-8)

print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print(f"F1 Score:      {f1_score:.4f}")
print(f"Dice Score:    {dice_score:.4f}")
print(f"IoU Score:     {iou_score:.4f}")