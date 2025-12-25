import torch
from torchvision import transforms

from dataset import get_train_val_loaders_from_txt
from helpers import train_model, eval_model, BCEDiceLoss
from model import UNet

from copy import deepcopy

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_size_h, img_size_w = 512, 512
    transform = transforms.Compose([
        transforms.Resize((img_size_h, img_size_w)),
        transforms.ToTensor()
    ])

    train_loader, val_loader, test_loader = get_train_val_loaders_from_txt('dataset/kneeMRI-ACL', transform)

    criterion = BCEDiceLoss()
    epochs = 10

    configs = [
        {"optimizer_type": "adam", "lr": 1e-4},
    ]

    for i, cfg in enumerate(configs):
        print(f"\n=== Training Config {i+1}: {cfg} ===")
        model = UNet(in_channels=1, out_channels=1)  # fresh model
        best_model_path, train_losses, val_losses = train_model(deepcopy(model), train_loader, val_loader, device, criterion,
                    optimizer_type=cfg["optimizer_type"], lr=cfg["lr"], epochs=epochs)


    model = UNet(in_channels=1, out_channels=1)

    # Load the weights
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    dice_avg, iou_avg, f1_avg, recall_avg, precision_avg = eval_model(model, test_loader, device)

    print(f"📊 Test Metrics:")
    print(f" - Dice Score: {dice_avg:.4f}")
    print(f" - IoU       : {iou_avg:.4f}")
    print(f" - F1 Score  : {f1_avg:.4f}")
    print(f" - Recall    : {recall_avg:.4f}")
    print(f" - Precision : {precision_avg:.4f}")


if __name__ == '__main__':
    main()