import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
import json

from tqdm import tqdm

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.contiguous()
        targets = targets.contiguous()

        intersection = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        return dice_loss.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0.0, path='best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience} without improvement")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f"Validation loss improved. Saving model to {self.path}")


def train_model(
        model, train_loader, val_loader, device, criterion, optimizer_type='adam', lr=1e-4,
        epochs=200, model_name_prefix="unet"):

    model.to(device)

    train_losses = []
    val_losses = []

    best_val_loss = 999
    best_epoch = -1

    # Select optimizer
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    cfg_str = f"{model_name_prefix}_{optimizer_type}_lr{lr:.0e}"
    save_dir = Path('results')
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{cfg_str}.pth"
    json_path = save_dir / f"{cfg_str}_metrics.json"

    early_stopping = EarlyStopping(patience=10, path=str(save_path))

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                loss = criterion(preds, masks)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f} \n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # === Save metrics to JSON ===
    metrics = {
        "optimizer": optimizer_type,
        "learning_rate": lr,
        "best_val_loss": round(best_val_loss, 6),
        "best_epoch": best_epoch,
        "model_path": str(save_path),
        "train_losses": train_losses,
        "val_losses": val_losses
    }

    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Best model saved to: {save_path}")
    print(f"Metrics saved to:    {json_path}")
    print(f"Best epoch: {best_epoch}, Best Val Loss: {best_val_loss:.6f}")
    print(f"Best config: optimizer={optimizer_type}, lr={lr:.0e}")

    return str(save_path), train_losses, val_losses



def compute_metrics(preds, targets, threshold=0.5):
        """Compute segmentation metrics for batch (assumes preds and targets are tensors)."""
        preds = (preds > threshold).float()
        targets = targets.float()

        smooth = 1e-5

        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)

        iou = (intersection + smooth) / ((preds + targets - preds * targets).sum(dim=(1, 2, 3)) + smooth)
        recall = (intersection + smooth) / (targets.sum(dim=(1, 2, 3)) + smooth)
        precision = (intersection + smooth) / (preds.sum(dim=(1, 2, 3)) + smooth)
        f1 = (2 * precision * recall) / (precision + recall + smooth)

        return dice.mean().item(), iou.mean().item(), f1.mean().item(), recall.mean().item(), precision.mean().item()

def eval_model(model, test_loader, device):
    model.eval()
    dice_total, iou_total, f1_total, recall_total, precision_total = 0, 0, 0, 0, 0
    num_batches = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)

            dice, iou, f1, recall, precision = compute_metrics(preds, masks)
            dice_total += dice
            iou_total += iou
            f1_total += f1
            recall_total += recall
            precision_total += precision
            num_batches += 1

    dice_avg = dice_total / num_batches
    iou_avg = iou_total / num_batches
    f1_avg = f1_total / num_batches
    recall_avg = recall_total / num_batches
    precision_avg = precision_total / num_batches

    return dice_avg, iou_avg, f1_avg, recall_avg, precision_avg