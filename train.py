import os, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import build_model
from metrics_manager import MetricsManager

CACHE_DIR = 'cache'
IMG_SIZE = 150
DATASET_DIR = '/home/ubuntu/ImageDataset' if os.name == 'posix' else r'K:\ImageDataset'   # Linux-friendly path for EC2

os.makedirs(CACHE_DIR, exist_ok=True)

_DEVICE_PRINTED = False

def print_gpu_info():
    """Prints GPU status information exactly once per session."""
    global _DEVICE_PRINTED
    if _DEVICE_PRINTED:
        return
    
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count > 1:
            print(f"✅ Found {count} GPUs! (Using DataParallel mode)")
        else:
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"⚠️ GPU not detected! Torch version: {torch.__version__} | CUDA version: {torch.version.cuda}")
        print("💡 TIP: Check if you have installed the CUDA-compatible version of PyTorch.")
    
    _DEVICE_PRINTED = True

def get_device(verbose=True):
    if verbose:
        print_gpu_info()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model_pipeline(model_name, epochs=5, batch_size=32, dataset_choice='yonsei_faces', data_root=DATASET_DIR):
    device = get_device(verbose=True)
    metrics = MetricsManager()

    print(f"🚀 Training {model_name} on {device}...")

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = os.path.join(data_root, dataset_choice)
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(model_name).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    metrics.start_timer()
    best_acc = 0.0
    save_path = os.path.join(CACHE_DIR, f"{model_name}_best.pth")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2%}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    duration = metrics.stop_timer()
    metrics.save_training_stats(model_name, duration, best_acc, epochs, dataset_choice)
    print(f"✅ Training finished! Best Accuracy: {best_acc:.2%} | Saved: {save_path}")
    return save_path, best_acc, duration

if __name__ == "__main__":
    train_model_pipeline("DenseNet121", epochs=5, batch_size=32)