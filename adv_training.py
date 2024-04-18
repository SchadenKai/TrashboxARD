import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import torch.optim as optim
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchattacks
import local_models 

# Device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparams
mean = torch.tensor([0.6521, 0.6325, 0.6088])
std = torch.tensor([0.2209, 0.2221, 0.2288])

input_size = 12288 # 3*64*64
height = 64
num_classes = 7
epochs = 100
lr = 0.01
lr_factor = 0.1
lr_schedule = [50,100, 150]
weight_decay = 0.0001
batch_size = 128
val_period = 1
file_name = "inceptionv3_trashbox_adv_training"
model_name = "Inceptionv3"

# Attack hyperparams 

epsilon = 8.0 / height
alpha = 2.0 / height
steps = 10
step_size = 0.003
beta = 1.0
distance='l_inf'

# Graph writer initialize
writer = SummaryWriter("runs/trashbox/" + file_name)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((height, height)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((height, height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
}

# Adjusting learning rate
def adjust_learning_rate(optimizer, epoch, lr):
    if epoch in lr_schedule:
        lr *= lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

print('===>> Preparing data...')
trash_train_dataset = torchvision.datasets.ImageFolder('dataset/trashbox/train', transform=data_transforms['train'])
trash_train_loader = torch.utils.data.DataLoader(dataset=trash_train_dataset, shuffle=True, batch_size=batch_size)
trash_val_dataset = torchvision.datasets.ImageFolder('dataset/trashbox/val', transform=data_transforms['val'])
trash_val_loader = torch.utils.data.DataLoader(dataset=trash_val_dataset, shuffle=True, batch_size=batch_size)

examples = iter(trash_val_loader)
samples,labels = next(examples)
img_grid = torchvision.utils.make_grid(samples)
writer.add_image("Trashbox images", img_grid)

print('====>> Setting up model...')
# model = local_models.xception(num_classes=num_classes).to(device)
model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
infeatures = model.fc.in_features
model.aux_logits = False

# Freeze layers in model 
for parameter in model.parameters():
    parameter.requires_grad = False
    
model.fc = nn.Linear(infeatures, num_classes, True, device)




attack = torchattacks.PGD(model=model, eps=epsilon, alpha=alpha, steps=steps)
criterion = nn.CrossEntropyLoss()

# # Initialize graph
# writer.add_graph(model, samples)
# writer.close()

def train(epoch, optimizer):
    train_loss = 0
    correct = 0
    total = 0
    model.train()
    iterator = tqdm(trash_train_loader, ncols=0, leave=False)
    for i, (inputs, targets)in enumerate(iterator):
        inputs, targets = inputs.to(device),targets.to(device)
        
        optimizer.zero_grad()
        adv_image = attack(inputs, targets)
        adv_output = model(adv_image)
        loss = criterion(adv_output, targets)
        loss.backward()
    
        optimizer.step()
        train_loss += loss.item()
        _, predicted = adv_output.max(1)
        
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if i % 10 == 0:
            print('\nCurrent batch:', str(i))
            print('Current adversarial train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current adversarial train loss:', loss.item())
    print('\nTotal adversarial train accuarcy:', 100. * correct / total)
    print('Total adversarial train loss:', train_loss)
    
    writer.add_scalar('Adversarial train loss: ' + model_name, train_loss)
    writer.add_scalar('Adversarial train accuracy: ' + model_name, 100. * correct / total)

def test(epoch, optimizer):
    print('\n[ Test epoch: %d ]' % epoch)
    model.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(trash_val_loader, ncols=0, leave=False)
        for i, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            
            print('normal input shape: ', inputs.shape)
            output = model(inputs)
            loss = criterion(output, targets)
            benign_loss += loss.item()

            _, predicted = output.max(1)
            benign_correct += predicted.eq(targets).sum().item()

            if i % 10 == 0:
                print('\nCurrent batch:', str(i))
                print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current benign test loss:', loss.item())
        
            adv = attack(inputs, targets)
            adv_outputs = model(adv)
            loss = criterion(adv_outputs, targets)
            adv_loss += loss.item()

            _, predicted = adv_outputs.max(1)
            adv_correct += predicted.eq(targets).sum().item()

            if i % 10 == 0:
                print('Current adversarial test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current adversarial test loss:', loss.item())
    print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
    print('Total adversarial test Accuarcy:', 100. * adv_correct / total)
    print('Total benign test loss:', benign_loss)
    print('Total adversarial test loss:', adv_loss)
    
    # Graph
    writer.add_scalar("Natural test accuracy: " + model_name, 100. * benign_correct / total)
    writer.add_scalar("Natural test loss: " + model_name, benign_loss)
    writer.add_scalar("Adversarial test accuracy: " + model_name, 100. * adv_correct / total)
    writer.add_scalar("Adversarial test loss: " + model_name, adv_loss)
    
    # Save checkpoint
    state = {
        'epoch' : epoch,
        'net': model.state_dict(),
        'optim' : optimizer.state_dict(),
        'loss' : loss
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + file_name)
    print('Model Saved!')
    return 

def main():
    learning_rate = lr
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, learning_rate)
        train_loss = train(epoch, optimizer)
        if (epoch + 1) % val_period == 0:
            test(epoch, optimizer)

if __name__ == '__main__':
    main()