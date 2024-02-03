import torch
# import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import equalize
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


num_classes = 9

# Define the transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda img: equalize(img)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model_path = "trained_models/resnet50.pth"

# load the model
resnet50 = torch.load(model_path)
# set the model to evaluation mode
resnet50.eval()

# Move the model to the GPU device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print the device that is being used
print(f"Using device: {device}")

# Create a data loader for the testing set
test_dataset = ImageFolder(root="data/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create a dictionary to map class names to class indices
class_to_idx = test_dataset.class_to_idx

# Create a list of class names
classes = [name for name in class_to_idx.keys()]

# Get the ground truth
ground_truth = []
for _, labels in test_loader:
    ground_truth.extend(labels.numpy())

# get the probabilities for each class for each image and store them in a csv file
probs = []
preds = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = resnet50(inputs)
        # get the probabilities for each class for each image
        probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())
        # get the class with the highest probability 
        preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

# format the probabilities values to float with 4 decimal places
probs = np.round(probs, 2)

# convert to a dataframe
probs = pd.DataFrame(probs, columns=classes)

# add the ground truth and predictions to the dataframe
probs["ground_truth"] = ground_truth
probs["predictions"] = preds

# label the predictions with the class names
probs["predictions"] = probs["predictions"].apply(lambda x: classes[x])
# label the ground truth with the class names
probs["ground_truth"] = probs["ground_truth"].apply(lambda x: classes[x])

# reorder the columns so that the ground truth and predictions are the first columns
cols = probs.columns.tolist()
cols = cols[-2:] + cols[:-2]
probs = probs[cols]

# save the probabilities in a csv file. Keep numbers to 2 decimal places
probs.to_csv("test_results/probs_resnet50.csv", index=False, float_format="%.2f")





