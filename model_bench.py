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
lr_schedule = [100,150]
lr_factor = 0.1
epochs = 100
output = ''
temp = 30.00
val_period = 1
save_period = 1
alpha = 1
dataset = 'Trashbox'
model = 'Mobilenetv3'
teacher_model = 'InceptionV3'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        rgba_to_tensor,
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        rgba_to_tensor,
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        rgba_to_tensor,
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print('==> Preparing data..'+ dataset)
dataset_path = 'dataset/trashbox/'

train_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'train'), data_transforms['train'])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=40, shuffle=True, num_workers=4)
val_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'val'), data_transforms['val'])
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=40, shuffle=False, num_workers=4)
test_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'test'), data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=40, shuffle=False, num_workers=4)
class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Class names: {class_names} Num classes: {num_classes}")

print('==> Building student model..'+ model)
basic_net = models.googlenet(num_classes=num_classes)
basic_net = basic_net.to(device)


KL_loss = nn.KLDivLoss()
XENT_loss = nn.CrossEntropyLoss()
learning_rate = lr
def train(epoch, optimizer):
    train_loss = 0
    iterator = tqdm(train_loader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        inputs, targets = inputs.to(device), targets.to(device)
        basic_outputs = basic_net(inputs)
        loss = XENT_loss(basic_outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        iterator.set_description(str(loss.item()))
    if (epoch+1)%save_period == 0:
        state = {
            'net': basic_net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint/'+dataset+'/'+model+'/'+output+'/'):
            os.makedirs('checkpoint/'+dataset+'/'+model+'/'+output+'/', )
        torch.save(state, './checkpoint/'+dataset+'/'+model+'/'+output+'/epoch='+str(epoch)+'.t7')
    print('Mean Training Loss:', train_loss/len(iterator))
    return train_loss

def test(epoch, optimizer):
    basic_net.eval()
    natural_correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(val_loader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            natural_outputs = basic_net(inputs)
            _, natural_predicted = natural_outputs.max(1)
            natural_correct += natural_predicted.eq(targets).sum().item()
            total += targets.size(0)
    natural_acc = 100.*natural_correct/total
    print('Natural acc:', natural_acc)
    return natural_acc

def test_final(epoch, optimizer):
    basic_net.eval()
    natural_correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(test_loader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            natural_outputs = basic_net(inputs)
            _, natural_predicted = natural_outputs.max(1)
            natural_correct += natural_predicted.eq(targets).sum().item()
            total += targets.size(0)
    natural_acc = 100.*natural_correct/total
    print('Natural acc:', natural_acc)
    return natural_acc

def main():
    # learning_rate = lr
    optimizer = optim.SGD(basic_net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-4)
    # for epoch in range(epochs):
    #     train_loss = train(epoch, optimizer)
    #     print('Train Loss: ', train_loss)
    #     if (epoch+ 1) % val_period == 0:
    #         natural_val = test(epoch, optimizer)
    #         print(f'Epoch: {epoch+1}, Natural Acc: {natural_val}')
    checkpoint = torch.load ('./checkpoint/Trashbox/googlenet/epoch=99.t7')
    basic_net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epochs = 100
    test_acc = test_final(epochs, optimizer)
    print('Test accuracy: ' + test_acc)



if __name__ == '__main__':
    print('testing if this runs first')
    main()