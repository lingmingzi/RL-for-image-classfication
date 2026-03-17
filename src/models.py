import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50


def build_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "resnet50":
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")
