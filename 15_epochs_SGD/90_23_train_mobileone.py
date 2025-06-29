import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Import MobileOne model
try:
    from mobileone import mobileone
    MobileOne = mobileone
except ImportError:
    raise ImportError("Could not import mobileone module. Ensure mobileone.py is in the correct directory.")

# Setup device-agnostic code
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# Define paths
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = Path(os.path.expanduser('~/Desktop/mobilenetv4.pytorch/mobileone_s4_unfused.pth.tar'))
TRAIN_DIR = Path(os.path.expanduser("~/Downloads/MergedSC_split/train"))
TEST_DIR = Path(os.path.expanduser("~/Downloads/MergedSC_split/test"))
VAL_DIR = Path(os.path.expanduser("~/Downloads/MergedSC_split/val"))

# Validate paths
for path in [CHECKPOINT_PATH, TRAIN_DIR, TEST_DIR, VAL_DIR]:
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=test_transform)

# Get class names and number of classes
class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Classes: {class_names}")
print(f"Number of classes: {num_classes}")

# Create data loaders
BATCH_SIZE = 16
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of val dataloader: {len(val_dataloader)} batches of {BATCH_SIZE}")

# Model setup
def create_mobileone_model(num_classes, checkpoint_path=None, variant='s4', device='cpu'):
    """Create MobileOne model with optional checkpoint loading"""
    if checkpoint_path and checkpoint_path.exists():
        print(f"Loading checkpoint from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

            # Create model with ImageNet classes to load pretrained weights
            print("Creating model with ImageNet structure to load pretrained weights...")
            pretrained_model = MobileOne(num_classes=1000, variant=variant)
            pretrained_dict = {k: v for k, v in state_dict.items() if not k.startswith('linear')}
            model_dict = pretrained_model.state_dict()
            model_dict.update(pretrained_dict)
            pretrained_model.load_state_dict(model_dict)
            print("Pretrained weights loaded successfully")

            # Create final model with correct number of classes
            model = MobileOne(num_classes=num_classes, variant=variant)
            pretrained_state = pretrained_model.state_dict()
            model_state = model.state_dict()

            for name, param in pretrained_state.items():
                if not name.startswith('linear'):
                    if name in model_state and param.shape == model_state[name].shape:
                        model_state[name] = param
                    else:
                        print(f"Skipping layer {name} due to shape mismatch")

            model.load_state_dict(model_state)
            print(f"Model created with {num_classes} classes using pretrained features")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Creating model from scratch.")
            model = MobileOne(num_classes=num_classes, variant=variant)
    else:
        model = MobileOne(num_classes=num_classes, variant=variant)
        print(f"Created model from scratch with {num_classes} classes")

    return model

# Create model
model = create_mobileone_model(num_classes=num_classes, checkpoint_path=CHECKPOINT_PATH, variant='s4', device=device)
model = model.to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train_step(model, dataloader, loss_fn, optimizer, device):
    """Train the model for one epoch with tqdm progress bar"""
    model.train()
    train_loss, train_acc = 0, 0
    loop = tqdm(dataloader, leave=False, desc='Training')
    
    for X, y in loop:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        acc = (y_pred_class == y).sum().item() / len(y_pred)
        train_acc += acc
        
        loop.set_postfix(loss=loss.item(), accuracy=acc)
        
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(model, dataloader, loss_fn, device):
    """Evaluate the model with tqdm progress bar"""
    model.eval()
    test_loss, test_acc = 0, 0
    loop = tqdm(dataloader, leave=False, desc='Testing')
    
    with torch.inference_mode():
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            test_pred_labels = test_pred_logits.argmax(dim=1)
            acc = (test_pred_labels == y).sum().item() / len(test_pred_labels)
            test_acc += acc
            
            loop.set_postfix(loss=loss.item(), accuracy=acc)
            
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc

def train_model(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device):
    best_test_loss = float('inf')
    best_test_acc = 0.0
    best_model_wts = None
    best_acc_model_wts = None

    train_loss_values, train_acc_values = [], []
    test_loss_values, test_acc_values = [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        
        scheduler.step()

        print(f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.5f} | Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.5f}")

        train_loss_values.append(train_loss)
        train_acc_values.append(train_acc)
        test_loss_values.append(test_loss)
        test_acc_values.append(test_acc)
        
        # Save best model by test loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, MODEL_PATH / "best_model_by_loss.pth")
            print(f"Saved best model by loss at epoch {epoch+1} with test loss {best_test_loss:.5f}")
        
        # Save best model by test accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_acc_model_wts = model.state_dict()
            torch.save(best_acc_model_wts, MODEL_PATH / "best_model_by_acc.pth")
            print(f"Saved best model by accuracy at epoch {epoch+1} with test acc {best_test_acc:.5f}")
            
    return {
        "train_loss": train_loss_values,
        "train_acc": train_acc_values,
        "test_loss": test_loss_values,
        "test_acc": test_acc_values
    }

# Set random seeds for reproducibility
torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed(42)

# Start training
print("Starting training...")
NUM_EPOCHS = 15

results = train_model(model=model,
                     train_dataloader=train_dataloader,
                     test_dataloader=test_dataloader,
                     optimizer=optimizer,
                     loss_fn=loss_fn,
                     epochs=NUM_EPOCHS,
                     device=device)

# Plot training curves
def plot_loss_curves(results):
    """Plot training and test loss/accuracy curves"""
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]
    epochs = range(len(results["train_loss"]))
    
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    
    plt.show()

plot_loss_curves(results)

# Save the trained model
MODEL_NAME = "mobileone_trained_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

# Load and test the saved model
def load_model(model_path, num_classes, variant='s4'):
    """Load a saved model"""
    loaded_model = MobileOne(num_classes=num_classes, variant=variant)
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    return loaded_model

# Test loading
loaded_model = load_model(MODEL_SAVE_PATH, num_classes, variant='s4')
loaded_model = loaded_model.to(device)

# Evaluate on validation set
val_loss, val_acc = test_step(loaded_model, val_dataloader, loss_fn, device)
print(f"Validation Loss: {val_loss:.5f}, Validation Accuracy: {val_acc:.5f}")

# Make predictions on a sample
def make_predictions(model, dataloader, device):
    """Make predictions and return results"""
    model.eval()
    predictions = []
    labels = []
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1)).item()
            predictions.extend(y_pred_class.cpu().numpy())
            labels.extend(y.cpu().numpy())
    
    return predictions, labels

# Get predictions on test set
test_predictions, test_labels = make_predictions(loaded_model, test_dataloader, device)

print(f"Made {len(test_predictions)} predictions")
print(f"Test accuracy: {(np.array(test_predictions) == np.array(test_labels)).mean():.4f}")