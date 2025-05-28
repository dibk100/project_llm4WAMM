import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights

def get_model(model_name: str, num_classes: int = 6):
    """
    모델 이름에 따라 모델 객체를 반환

    Args:
        model_name (str): 'resnet18', 'resnet50', 'custom_resnet50'
        num_classes (int): 출력 클래스 수

    Returns:
        nn.Module: 선택한 모델 객체
    """
    model_name = model_name.lower()
    
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True,weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'custom_resnet50':
        model = CustomResNet50(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model
