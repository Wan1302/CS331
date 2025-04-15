import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MiniLocalizationNet(nn.Module):
    def __init__(self):
        super(MiniLocalizationNet, self).__init__()

        self.conv = nn.Sequential(
            # Input: 450x450x3
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),   # -> 450x450x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # -> 225x225x16

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # -> 225x225x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # -> 112x112x32

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # -> 112x112x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # -> 56x56x64

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # -> 56x56x128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # -> 28x28x128

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),# -> 28x28x256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # -> 14x14x256

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),# -> 14x14x512
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),                  # -> 13x13x512

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),# -> 13x13x1024
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # -> 1x1x1024

        self.fc = nn.Sequential(
            nn.Flatten(),                          # -> 1024
            nn.Dropout(0.5),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 4)                     # Output: 4 bounding box values
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniLocalizationNet()  
state_dict = torch.load("model.pth", map_location=device) 
model.load_state_dict(state_dict)
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((450, 450)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở webcam.")
    exit()

num_photos = 10
captured = 0
image_paths = []

while captured < num_photos:
    ret, frame = cap.read()
    cropped = frame[120:120+600, 200:200+600]
    cv2.imshow("Nhấn Enter để chụp ảnh...", cropped)

    if cv2.waitKey(1) & 0xFF == 13:  # Phím Enter
        save_path = f"demo/image_{captured+1}.jpg"
        cv2.imwrite(save_path, cropped)
        print(f"Đã lưu {save_path}")
        image_paths.append(save_path)
        captured += 1

cap.release()
cv2.destroyAllWindows()

for i, image_path in enumerate(image_paths):
    pil_img = Image.open(image_path).convert("RGB")
    orig_width, orig_height = pil_img.size
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_bbox = model(input_tensor)[0].cpu().numpy()

    print(f"pred_bbox: {pred_bbox}") 

    x_center, y_center, w, h = pred_bbox
    x_center *= orig_width
    y_center *= orig_height
    w *= orig_width
    h *= orig_height

    x = x_center - w / 2
    y = y_center - h / 2

    fig, ax = plt.subplots()
    ax.imshow(pil_img)

    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.axis("off")
    plt.title(f"Ảnh {i+1} với bbox")

    save_path = f"demo/image_{i+1}_bbox.jpg"
    fig.savefig(save_path, bbox_inches='tight')
    print(f"Đã lưu {save_path}")
    plt.close()

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, ax in enumerate(axes.flat):
    if i < len(image_paths):
        img = Image.open(f"demo/image_{i+1}_bbox.jpg")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Ảnh {i+1}")

plt.tight_layout()
plt.show()