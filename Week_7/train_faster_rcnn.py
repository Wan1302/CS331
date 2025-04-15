import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# ========== 1. CUSTOM DATASET FOR YOLO LABEL FORMAT ==========
class YoloDataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms=None, class_names=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transforms = transforms
        self.class_names = class_names or []

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace(".jpg", ".txt"))
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    xmin = (x_center - width / 2) * w
                    ymin = (y_center - height / 2) * h
                    xmax = (x_center + width / 2) * w
                    ymax = (y_center + height / 2) * h
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(int(class_id))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

# ========== 2. COLLATE FUNCTION ==========
def collate_fn(batch):
    return tuple(zip(*batch))

# ========== 3. TRAINING FUNCTION ==========
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    for images, targets in pbar:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        pbar.set_postfix({k: f"{v.item():.4f}" for k, v in loss_dict.items()})

# ========== 4. EVALUATE MAP@50 ==========
def evaluate_map50(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            preds = model(images)
            
            formatted_preds = []
            for pred in preds:
                formatted_preds.append({
                    'boxes': pred['boxes'],
                    'scores': pred['scores'],
                    'labels': pred['labels']
                })
            
            all_preds.extend(formatted_preds)
            all_targets.extend(targets)
    
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True, iou_type="bbox", iou_thresholds=[0.5])
    metric.update(all_preds, all_targets)
    results = metric.compute()
    
    map50 = results["map_50"].item()
    return map50
    
# ========== 5. MAIN FUNCTION ==========
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = T.Compose([
        T.ToTensor()
    ])

    train_dataset = YoloDataset(
        image_dir=os.path.join(args.data_dir, 'images/train'),
        label_dir=os.path.join(args.data_dir, 'labels/train'),
        transforms=transforms
    )
    val_dataset = YoloDataset(
        image_dir=os.path.join(args.data_dir, 'images/val'),
        label_dir=os.path.join(args.data_dir, 'labels/val'),
        transforms=transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, args.num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        map50 = evaluate_map50(model, val_loader, device)
        print(f"[Epoch {epoch+1}] mAP@50: {map50:.4f}")
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")

    print("Training completed.")

# ========== 6. ARGUMENT PARSER ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset-traffic", help="Path to dataset folder")
    parser.add_argument("--num_classes", type=int, default=2, help="Include background as class 0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    args = parser.parse_args()

    main(args)
