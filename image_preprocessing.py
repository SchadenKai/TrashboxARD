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
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter('runs/prep/trashbox')

input_size = 12288 # 3*64*64
height = 64
num_classes= 7
batch_size = 128

mean = 0.5630
std = 0.3205

data_transforms = transforms.Compose([
    transforms.Resize((height, height)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std])
])

horizontal_transforms = transforms.Compose([
    transforms.Resize((height, height)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

print("====>> Preparing data...")
train_dataset = torchvision.datasets.ImageFolder('dataset/trashbox/train', transform=data_transforms)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size)

examples = iter(train_dataloader)
samples, labels = next(examples)
img_grid = torchvision.utils.make_grid(samples)
writer.add_image("Trashbox images after normalization", img_grid)