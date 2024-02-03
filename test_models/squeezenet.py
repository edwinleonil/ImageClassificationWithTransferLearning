import torch
from PIL import Image
import torchvision.transforms as transforms


# Save the model and weights
model_path = "pretrained_models/squeezenet1_0.pth"

# load the model
squeezenet = torch.load(model_path)
# set the model to evaluation mode
squeezenet.eval()

# load the image
img_path = "images/dog.jpg"
img = Image.open(img_path)

# define the transforms
transform = transforms.Compose([
    transforms.Resize(224),  # resize the image to 224x224 pixels
    transforms.CenterCrop(224),  # crop the image to 224x224 pixels about the center
    transforms.ToTensor(),  # convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize the image
])

# apply the transforms
img_transformed = transform(img)

# unsqueeze the image
img_transformed = img_transformed.unsqueeze(0)

# pass the image through the model
output = squeezenet(img_transformed)

# get the class names from txt file
with open("test_models/imagenet-classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# get the index of the highest output
prediction = output.argmax(dim=1).item()
print(f"Prediction: {classes[prediction]}")