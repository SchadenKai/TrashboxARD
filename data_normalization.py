import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
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


data_transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor()
])
print('===>> Preparing data...')
trash_train_dataset = torchvision.datasets.ImageFolder('dataset/trashbox/train', transform=data_transforms)
trash_train_loader = torch.utils.data.DataLoader(dataset=trash_train_dataset, shuffle=False, batch_size=1)

mean = 0.
std = 0.
nb_samples = 0.
for data, _ in trash_train_loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(mean)
print(std)