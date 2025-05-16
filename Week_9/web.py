import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from torchvision import transforms
from PIL import Image


def set_seed(seed=42):
    random.seed(seed)                    
    np.random.seed(seed)                 
    torch.manual_seed(seed)              
    torch.cuda.manual_seed(seed)         
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class OCRDataset:
    def __init__(self, img_width, img_height):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor()
        ])

    def __call__(self, image):
        image = self.transform(image)
        image = image.permute(0, 2, 1)  # (C, W, H)
        return image

class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 32, 200, 50)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # (B, 32, 100, 25)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (B, 64, 100, 25)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # (B, 64, 50, 12)
        )

        self.linear1 = nn.Linear(64 * 12, 64)

        self.lstm1 = nn.LSTM(64, 128, bidirectional=True, batch_first=True)  # (B, T, 128)
        self.dropout1 = nn.Dropout(0.25)
        self.lstm2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True)  # (B, T, 128)
        self.dropout2 = nn.Dropout(0.25)

        self.classifier = nn.Linear(128, num_classes)  # 64 * 2 (bi-directional)

    def forward(self, x):
        x = self.cnn(x)                   # (B, C, W, H) = (B, 64, 50, 12)
        x = x.permute(0, 2, 3, 1)         # (B, W, H, C) = (B, 50, 12, 64)
        x = x.reshape(x.size(0), x.size(1), -1) # (B, W, H * C) = (B, 50, 768)
        x = self.linear1(x)              # (B, W, 64)
        x = nn.Dropout(0.2)(x)            # (B, W, 64)

        x, _ = self.lstm1(x)             # (B, W, 256)
        x = self.dropout1(x)              # (B, W, 256)
        x, _ = self.lstm2(x)             # (B, W, 128)
        x = self.dropout2(x)              # (B, W, 128)

        x = self.classifier(x)           # (B, W, num_classes)
        x = F.log_softmax(x, dim=2)      # for CTC loss

        x = x.permute(1, 0, 2)           # (T, B, C)
        return x
    
def decode_ctc(output, num_to_char):
    out = output.permute(1, 0, 2)
    out_decoded = []
    for logits in out:
        pred = torch.argmax(logits, dim=-1)
        prev = -1
        decoded = []
        for p in pred:
            if p.item() != prev and p.item() != 0:
                decoded.append(num_to_char[p.item()])
            prev = p.item()
        out_decoded.append("".join(decoded))
    return out_decoded 

def load_model(model_path="ocr_model.pth", img_width=200, img_height=50):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = 'archive/samples'
    images = os.listdir(data_path)
    labels = [img.split(os.path.sep)[-1].split(".")[0] for img in images]
    characters = set(char for label in labels for char in label)
    characters = sorted(list(characters))
    characters = ["-"] + characters
    # Mapping characters to integers
    char_to_num = {char: idx for idx, char in enumerate(characters)}

    # Mapping integers back to original characters
    num_to_char = {idx: char for char, idx in char_to_num.items()}

    test_image = OCRDataset(img_width, img_height)

    model = OCRModel(num_classes=len(characters)).to(device)
    model.load_state_dict(torch.load("ocr_model.pth"))

    model.eval()
    return model, test_image, char_to_num, num_to_char
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.title("🧠 OCR Captcha Recognition with CRNN + CTC")
st.markdown("Upload a captcha image (e.g. `abcde.png`) to predict the characters.")

uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    filename = uploaded_file.name
    label_str = os.path.splitext(filename)[0]

    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption=f"Ground Truth: `{label_str}`", width=400)

    model, test_image, char_to_num, num_to_char = load_model()

    image_tensor = test_image(image_pil).unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        output = model(image_tensor)
        prediction = decode_ctc(output, num_to_char)[0]

    st.success(f"✅ Predicted: `{prediction}`")
    if prediction == label_str:
        st.markdown("🎉 Perfect Match!")
    else:
        st.markdown("🔍 Mismatch! Try another sample.")