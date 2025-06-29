import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from mobileone import mobileone  # Make sure mobileone supports variant='s4'
from collections import Counter
import cv2
import numpy as np
from PIL import Image

# --- Device Setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🔋 Using device: {device}")

torch.manual_seed(42)
# Median filter class using OpenCV
class MedianFilterTransform:
    def __init__(self, ksize=5):  # ksize must be odd and > 1
        self.ksize = ksize

    def __call__(self, img):
        # Convert PIL to OpenCV (BGR)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Apply OpenCV's built-in median filter
        filtered = cv2.medianBlur(img_cv, self.ksize)
        
        # Convert back to PIL (RGB)
        filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
        return Image.fromarray(filtered_rgb)

median_filter = MedianFilterTransform(ksize=5)

# --- Paths ---
MODEL_PATH = os.path.expanduser("/Users/dilli/Documents/Project_ICRTAI_2025/MobileOne/ICRTAI_MOBILEONE/Models/June29/best_model_by_test_acc.pth")
TEST_DIR = os.path.expanduser("/Users/dilli/Downloads/MergedSC_split/test") 
CLASS_DIR = os.path.expanduser("/Users/dilli/Downloads/MergedSC_split/test/malignant")  # Change to /benign as needed
TRUE_LABEL = 1  # 0 for benign, 1 for malignant

# --- Transforms ---
transform = transforms.Compose([
    median_filter,
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Dataset and Dataloader ---
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# --- Load Model (Correct Variant) ---
model = mobileone(num_classes=2, variant='s4')  # Use correct variant!
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("🌀 Model loaded and being evaluated...")

# --- Inference ---
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# --- Classification Report ---
report = classification_report(all_labels, all_preds, target_names=test_dataset.classes, digits=4)
print("📋 Classification Report:\n")
print(report)

# --- Overall Accuracy ---
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(all_labels, all_preds)
print(f"✅ Overall Accuracy: {accuracy:.4f}")

import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)  # raw logits
        
        # If model outputs logits for 2 classes:
        probs = F.softmax(outputs, dim=1)  # shape (batch_size, 2)
        
        preds = torch.argmax(probs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Now plot ROC curve using all_probs and all_labels
if all_probs.shape[1] == 1:
    # Binary classifier with single output logits - apply sigmoid
    probs = all_probs.ravel()  # flatten
else:
    # Two-class softmax output, take positive class prob (index 1)
    probs = all_probs[:, 1]

fpr, tpr, _ = roc_curve(all_labels, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("roc_curve.png", dpi=300, bbox_inches='tight')
plt.show()
