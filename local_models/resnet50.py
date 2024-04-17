import torch
import torch.nn as nn
import torchvision 
import torchvision.models as models

def resnet50(pretrained=True, num_classes=7):
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    print(model)
    return model
