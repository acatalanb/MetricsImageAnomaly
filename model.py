import torch
import torch.nn as nn
from torchvision import models

def build_model(model_name, pretrained=True):
    print(f"Building PyTorch model: {model_name}...")
    if model_name == "ResNet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 1)
        )
    elif model_name == "DenseNet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 1)
        )
    elif model_name == "EfficientNetB0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 1)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
        if model_name == "ResNet50":
            for param in model.fc.parameters(): param.requires_grad = True
        elif model_name == "DenseNet121":
            for param in model.classifier.parameters(): param.requires_grad = True
        elif model_name == "EfficientNetB0":
            for param in model.classifier[1].parameters(): param.requires_grad = True
    return model