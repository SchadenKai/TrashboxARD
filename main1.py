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
lr_schedule = [100,150]
lr_factor = 0.1
epochs = 100
output = ''
temp = 30
val_period = 1
save_period = 1
alpha = 1
dataset = 'Trashbox'
model = 'Mobilenetv3'
teacher_model = 'Resnet50'


# parser = argparse.ArgumentParser(description='Trashbox ARD Training')

# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--lr_schedule', type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')
# parser.add_argument('--lr_factor', default=0.1, type=float, help='factor by which to decrease lr')
# parser.add_argument('--epochs', default=100, type=int, help='number of epochs for training')
# parser.add_argument('--output', default = '', type=str, help='output subdirectory')
# parser.add_argument('--model', default = 'MobileNetV3', type = str, help = 'student model name')
# parser.add_argument('--teacher_model', default = 'Googlenet', type = str, help = 'teacher network model')
# parser.add_argument('--teacher_path', default = '', type=str, help='path of teacher net being distilled')
# parser.add_argument('--temp', default=30.0, type=float, help='temperature for distillation')
# parser.add_argument('--val_period', default=1, type=int, help='print every __ epoch')
# parser.add_argument('--save_period', default=1, type=int, help='save every __ epoch')
# parser.add_argument('--alpha', default=1.0, type=float, help='weight for sum of losses')
# parser.add_argument('--dataset', default = 'Trashbox', type=str, help='name of dataset')
# args = parser.parse_args()


# Set up device

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Learning rate

def adjust_learning_rate(optimizer, epoch, lr):
    if epoch in lr_schedule:
        lr *= lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# Define the path to your dataset

dataset_path = 'dataset/trashbox/'

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
}

# Set up dataset and dataloaders


print('==> Preparing data..'+ dataset)

# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
# ])
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
# ])
# if dataset == 'CIFAR10':
#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
#     testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
#     num_classes = 10
# elif dataset == 'CIFAR100':
#     trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
#     testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
#     num_classes = 100
# elif dataset == 'Trashbox':
#     train_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'train'), data_transforms['train'])
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
#     val_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'val'), data_transforms['val'])
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)
#     class_names = train_dataset.classes
#     print(f"Class names: {class_names}")
#     num_classes = 7

train_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'train'), data_transforms['train'])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'val'), data_transforms['val'])
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)
class_names = train_dataset.classes
print(f"Class names: {class_names}")
num_classes = 7

# Set student model

print('==> Building student model..'+ model)
# if model == 'Googlenet':
# 	basic_net = models.googlenet(pretrained=False, num_classes=num_classes)
# elif model == 'InceptionV3':
# 	basic_net = models.inception_v3(pretrained=False, num_classes=num_classes)
# elif model == 'ResNet50':
# 	basic_net = models.resnet50(pretrained=False, num_classes=num_classes)
# elif model == 'Xception':
# 	basic_net = xception(pretrained=False, num_classes=num_classes)
# elif model == 'MobileNetv3':
# 	model = models.mobilenet_v3_small(pretrained=False, num_classes=num_classes)
# mobilenet_v3_small_default_weights = models.MobileNet_V3_Small_Weights.DEFAULT
basic_net = models.mobilenet_v3_small(weights=None, num_classes=num_classes)
basic_net = basic_net.to(device)

# Set teacher model

print('==> Building teacher model..'+ teacher_model)
# if teacher_path != '':
# 	if teacher_model == 'Googlenet':
# 		teacher_net = models.googlenet(pretrained=False, num_classes=num_classes)
# 	elif teacher_model == 'InceptionV3':
# 		teacher_net = models.inception_v3(pretrained=False, num_classes=num_classes)
# 	elif teacher_model == 'ResNet50':
# 		teacher_net = models.resnet50(pretrained=False, num_classes=num_classes)
# 	elif teacher_model == 'Xception':
# 		teacher_net = xception(pretrained=False, num_classes=num_classes)

# 	teacher_net = teacher_net.to(device)
# 	for param in teacher_net.parameters():
# 		param.requires_grad = False

teacher_net = models.resnet50(weights=None, num_classes=num_classes)
teacher_net = teacher_net.to(device)
for param in teacher_net.parameters():
    param.requires_grad = False

# Hyperparameters

config = {
    'epsilon': 8.0 / 255,
    'num_steps': 10,
    'step_size': 2.0 / 255,
}

# Setup Adversarial Attack

class AttackPGD(nn.Module):
    def __init__(self, basic_net, config):
        super(AttackPGD, self).__init__()
        self.basic_net = basic_net
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']

    def forward(self, inputs, targets):
        x = inputs.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(self.basic_net(x), targets, reduction='sum')
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size*torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0.0, 1.0)
        return self.basic_net(x), x
    
# Initilize Attack

net = AttackPGD(basic_net, config)
if device == 'cuda':
    cudnn.benchmark = True

# Setup loss functions

KL_loss = nn.KLDivLoss()
XENT_loss = nn.CrossEntropyLoss()
learning_rate = lr

# Model training function

def train(epoch, optimizer):
    net.train()
    train_loss = 0
    iterator = tqdm(train_loader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, pert_inputs = net(inputs, targets)
        teacher_outputs = teacher_net(inputs)
        basic_outputs = basic_net(inputs)
        loss = alpha*temp*temp*KL_loss(F.log_softmax(outputs/temp, dim=1),F.softmax(teacher_outputs/temp, dim=1))+(1.0-alpha)*XENT_loss(basic_outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        iterator.set_description(str(loss.item()))
    if (epoch+1)%save_period == 0:
        state = {
            'net': basic_net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint/'+dataset+'/'+output+'/'):
            os.makedirs('checkpoint/'+dataset+'/'+output+'/', )
        torch.save(state, './checkpoint/'+dataset+'/'+output+'/epoch='+str(epoch)+'.t7')
    print('Mean Training Loss:', train_loss/len(iterator))
    return train_loss

# Model Testing function

def test(epoch, optimizer):
    net.eval()
    adv_correct = 0
    natural_correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(val_loader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            adv_outputs, pert_inputs = net(inputs, targets)
            natural_outputs = basic_net(inputs)
            _, adv_predicted = adv_outputs.max(1)
            _, natural_predicted = natural_outputs.max(1)
            natural_correct += natural_predicted.eq(targets).sum().item()
            total += targets.size(0)
            adv_correct += adv_predicted.eq(targets).sum().item()
            iterator.set_description(str(adv_predicted.eq(targets).sum().item()/targets.size(0)))
    robust_acc = 100.*adv_correct/total
    natural_acc = 100.*natural_correct/total
    print('Natural acc:', natural_acc)
    print('Robust acc:', robust_acc)
    return natural_acc, robust_acc


# Main function

def main():
    learning_rate = lr
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-4)
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, learning_rate)
        train_loss = train(epoch, optimizer)
        print('Train Loss: ', train_loss)
        if (epoch+ 1) % val_period == 0:
            natural_val, robust_val = test(epoch, optimizer)
            print(f'Epoch: {epoch+1}, Natural Acc: {natural_val}, Robust Acc: {robust_val}')

if __name__ == '__main__':
    main()