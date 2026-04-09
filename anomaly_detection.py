import torch
import os
from PIL import Image
from model import build_model
from metrics_manager import MetricsManager
from torchvision import transforms
from train import train_model_pipeline, get_device
import numpy as np

CACHE_DIR = 'cache'
DATASET_DIR = r'K:\ImageDataset'
IMG_SIZE = 150
device = get_device(verbose=False)

def load_model(model_arch, model_path):
    model = build_model(model_arch, pretrained=False).to(device) # Build model architecture without pretrained weights
    state_dict = torch.load(model_path, map_location=device)
    
    # Check if the model was saved with DataParallel (keys start with 'module.')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    is_dp = any(k.startswith('module.') for k in state_dict.keys())
    
    if is_dp:
        print("ℹ️ Detected DataParallel state dict. Stripping 'module.' prefix.")
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
        
    model.eval()
    return model

def predict_image(model_path, image_path, model_arch):
    model = load_model(model_arch, model_path)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    label = "ANOMALY" if prob > 0.5 else "NORMAL"
    confidence = prob * 100 if label == "ANOMALY" else (1 - prob) * 100
    return label, confidence

def train(model, epochs, batch, dataset):
    # This function is not used directly in anomaly_detection.py after refactoring
    # The train_model_pipeline from train.py is called directly
    pass

def evaluate_test_set(model_path, dataset_choice='ucirvine_chest_xray', model_arch="DenseNet121"):
    model = load_model(model_arch, model_path)
    test_dir = os.path.join(DATASET_DIR, dataset_choice, 'test')
    metrics_mgr = MetricsManager()
    y_true, y_probs = [], []
    metrics_mgr.start_timer()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for label_idx, category in enumerate(['NORMAL', 'ANOMALY']):
        folder = os.path.join(test_dir, category)
        if not os.path.exists(folder): continue
        for fname in os.listdir(folder):
            try:
                img = Image.open(os.path.join(folder, fname)).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    prob = torch.sigmoid(model(tensor)).item()
                y_true.append(label_idx)
                y_probs.append(prob)
            except Exception:
                continue

    duration = metrics_mgr.stop_timer()
    if not y_true:
        print(f"❌ No images found in {test_dir}.")
        return None

    results = metrics_mgr.calculate_metrics(y_true, y_probs)

    # === NEW: Use dataset-specific ROC file ===
    # Modified to pass model_arch to plot_roc_curve
    roc_auc = metrics_mgr.plot_roc_curve(y_true, y_probs, dataset_name=dataset_choice, model_name=model_arch, save_dir=CACHE_DIR) # Modified line
    results["roc_auc"] = float(roc_auc)

    print(f"✅ Evaluation complete in {duration:.1f}s | AUC-ROC: {results['roc_auc']:.4f}")
    # Save confusion matrix with model and dataset name
    fig = metrics_mgr.plot_confusion_matrix(np.array(results['confusion_matrix']))
    fig.savefig(f"{CACHE_DIR}/confusion_matrix_{dataset_choice}_{model_arch}.png") # Modified line
    return results

# ====================== RUN MODES ======================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict", "evaluate"], required=True)
    parser.add_argument("--model", default="DenseNet121", choices=["DenseNet121", "ResNet50", "EfficientNetB0"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--dataset", default="ucirvine_chest_xray")
    parser.add_argument("--image", help="Path to image for prediction")
    parser.add_argument("--model_path", help="Path to .pth model")
    args = parser.parse_args()

    if args.mode == "train":
        train_model_pipeline(args.model, args.epochs, args.batch, args.dataset, data_root=DATASET_DIR)
    elif args.mode == "predict":
        if not args.image or not args.model_path:
            print("❌ Provide --image and --model_path")
        else:
            label, confidence = predict_image(args.model_path, args.image, args.model)
            print(f"RESULT: {label} ({confidence:.1f} %)")
    elif args.mode == "evaluate":
        if not args.model_path:
            print("❌ Provide --model_path")
        else:
            evaluate_test_set(args.model_path, args.dataset, args.model)