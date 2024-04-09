# Load dependencies

import torch
import torch.nn as nn
import torch.optim as optim
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
from local_models import *
from PIL import Image, UnidentifiedImageError

# Hyper params
lr = 0.1
lr_schedule = [50,100]
lr_factor = 0.1
epochs = 100
output = ''
temp = 30.00
val_period = 1
save_period = 1
alpha = 1

# Set up device

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.memory_summary(device=None, abbreviated=False)
torch.cuda.empty_cache()

# Learning rate

def adjust_learning_rate(optimizer, epoch, lr):
    if epoch in lr_schedule:
        lr *= lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


# Define data transformations

def rgba_to_tensor(image):
    try:
        if isinstance(image, Image.Image):
            if image.mode == 'RGBA':
                return TF.to_tensor(image.convert('RGB'))
            return TF.to_tensor(image.convert('RGB'))
        elif isinstance(image, torch.Tensor):
            return TF.to_tensor(TF.to_pil_image(image).convert('RGB'))
        else:
            raise TypeError(f'Input should be either a PIL Image or a PyTorch Tensor, but got {type(image)}')
    except UnidentifiedImageError:
        print(f"Skipping unidentified image: {image}")
        return None


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        rgba_to_tensor,
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        rgba_to_tensor,
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        rgba_to_tensor,
    ]),
}
# Set up dataset and dataloaders


print('==> Preparing data..')
dataset_path = 'dataset/trashbox/'
train_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'train'), data_transforms['train'])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=120, shuffle=True, num_workers=4)
test_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'val'), data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=120, shuffle=False, num_workers=4)
class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Class names: {class_names} Num classes: {num_classes}")

# Set teacher model

print('==> Building teacher model..')
teacher_net = models.googlenet(pretrained=True, num_classes=num_classes)
teacher_net = teacher_net.to(device)
for param in teacher_net.parameters():
    param.requires_grad = False

# Hyperparameters for adversarial attack

config = {
    'epsilon': 8.0 / 255,
    'num_steps': 10,
    'step_size': 2.0 / 255,
}

# Setup loss functions

XENT_loss = nn.CrossEntropyLoss()
learning_rate = lr

# Model training function


def train(epoch, optimizer):
    train_loss = 0
    iterator = tqdm(train_loader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        teacher_outputs = teacher_net(inputs).logits
        loss = XENT_loss(teacher_outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    if (epoch+1)%save_period == 0:
        state = {
            'net': teacher_net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint/Trashbox/Normal'+output+'/'):
            os.makedirs('checkpoint/Trashbox/Normal'+output+'/', )
        torch.save(state, './checkpoint/Trashbox/Normal'+output+'/epoch='+str(epoch)+'.t7')
    print('Mean Training Loss:', train_loss/len(iterator))
    return train_loss

# Model Testing function

def test(epoch, optimizer):
    natural_correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(test_loader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            natural_outputs = teacher_net(inputs)
            _, natural_predicted = natural_outputs.max(1)
            natural_correct += natural_predicted.eq(targets).sum().item()
            total += targets.size(0)
    natural_acc = 100.*natural_correct/total
    print('Natural acc:', natural_acc)
    return natural_acc


# Main function

def main():
    learning_rate = lr
    optimizer = optim.SGD(teacher_net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-4)
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, learning_rate)
        train_loss = train(epoch, optimizer)
        print('Train Loss: ', train_loss)
        if (epoch+ 1) % val_period == 0:
            natural_val, robust_val = test(epoch, optimizer)
            print(f'Epoch: {epoch+1}, Natural Acc: {natural_val}, Robust Acc: {robust_val}')

if __name__ == '__main__':
    main()