import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils
import torch.utils.data
import torchvision
import torchvision.models as models
import torch.optim as optim
import os
import argparse
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter("runs/trashbox/data_normalized_preview")

mean = torch.tensor([0.6521, 0.6325, 0.6088])
std = torch.tensor([0.2209, 0.2221, 0.2288])

# mean = torch.tensor([0.485, 0.456, 0.406])
# std = torch.tensor([0.229, 0.224, 0.225])

data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)
])
print('===>> Preparing data...')
trash_train_dataset = torchvision.datasets.ImageFolder('dataset/trashbox/train', transform=data_transforms)
trash_train_loader = torch.utils.data.DataLoader(dataset=trash_train_dataset, shuffle=False, batch_size=32)
# imagenet_dataset = torchvision.datasets.ImageNet(root="dataset/", train=True, download=True, transforms=data_transforms)
# imagenet_loader = torch.utils.data.DataLoader(dataset=imagenet_dataset, shuffle=False, batch_size=32)

examples = iter(trash_train_loader)
samples, labels = next(examples)
img_grid = torchvision.utils.make_grid(samples)
writer.add_image("Normalized Trashnet Image with Imagenet normalization values", img_grid)