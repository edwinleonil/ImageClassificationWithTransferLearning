import os
import torch
from torchvision import models
# get the weights from torchvision
from torchvision.models import GoogLeNet_Weights, ResNet18_Weights, ResNet50_Weights, ResNet101_Weights, MobileNet_V2_Weights, Inception_V3_Weights, SqueezeNet1_0_Weights

# Define the models to download
models_to_download = {
    "googlenet": models.googlenet,
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "mobilenet_v2": models.mobilenet_v2,
    "inception_v3": models.inception_v3,
    "squeezenet1_0": models.squeezenet1_0,
}

# list the weights to download
weights_to_download = {
    "googlenet": GoogLeNet_Weights,
    "resnet18": ResNet18_Weights,
    "resnet50": ResNet50_Weights,
    "resnet101": ResNet101_Weights,
    "mobilenet_v2": MobileNet_V2_Weights,
    "inception_v3": Inception_V3_Weights,
    "squeezenet1_0": SqueezeNet1_0_Weights,
}

# Directory to save the models
save_directory = "pretrained_models"

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Download and save the models using the weights defined above
for model_name, model_fn in models_to_download.items():
    # get the weights
    weights = weights_to_download[model_name]

    # get the model with pretrained weights
    model = model_fn(weights=weights.DEFAULT)

    # Save the model and weights
    save_path = os.path.join(save_directory, f"{model_name}.pth")
    torch.save(model, save_path)

    print(f"Saved {model_name} model and weights to: {save_path}")

