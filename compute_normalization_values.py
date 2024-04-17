import torch
from torchvision import datasets, transforms

# Define the transform to convert images to tensors
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load your dataset
dataset = datasets.ImageFolder('dataset/trashbox/train', transform=transform)

# Create a data loader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        print(images.shape)
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3)).sum()
        std += images.std(axis=(0, 2, 3)).sum()

    mean /= num_pixels
    std /= num_pixels

    return mean, std

mean, std = get_mean_std(data_loader)
print(f'Mean: {mean}')
print(f'STD: {std}')