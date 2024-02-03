import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torchvision.transforms.functional import equalize
from PIL import Image
import matplotlib.pyplot as plt

# 1. Load one image at a time and apply transforms
image = Image.open(r'C:\test_image.png')

# 2. Apply transforms
transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),  # Random reflection in the left-right direction
    v2.RandomVerticalFlip(),    # Random reflection in the up-down direction
    # add random scale between 0.8 and 1.2
    v2.RandomAffine(degrees=0, scale=(0.8, 1.2)),
    v2.Grayscale(num_output_channels=1),  # convert image to grayscale
    v2.Lambda(lambda img: equalize(img)),  # equalize image
    v2.ToTensor(),
    v2.Lambda(lambda img: img.repeat(3, 1, 1)),  # repeat grayscale image 3 times to get 3-channel grayscale
    v2.Normalize(mean=[0.5,0.5,0.5],   # adjusted for 3-channel grayscale
                std=[0.5,0.5,0.5])   # adjusted for 3-channel grayscale
])

image_transformed = transforms(image)

# 3. Plot original and transformed image
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original image')
plt.subplot(1, 2, 2)
plt.imshow(image_transformed.permute(1, 2, 0))
plt.title('Transformed image')
plt.show()