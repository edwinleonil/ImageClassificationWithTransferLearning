import torch
import timm
import os

# Define the models to download
models_to_download = {
    "xception": timm.models.xception,  # Xception from timm library https://timm.fast.ai/
    # "resnet50": timm.models.resnet50,  # ResNet50 from timm library https://timm.fast.ai/
    # "BiT-M-R50x1": timm.models.vision_transformer.BiT_M_R50x1  # BiT-M-R50x1 from timm library https://timm.fast.ai/
}

# Directory to save the models
save_directory = "pretrained_models"

# Create the directory if it doesn't exist

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Download and save the models
for model_name, model_fn in models_to_download.items():
    if model_name == "xception":
        model = model_fn(pretrained=True)
    else:
        model = model_fn(pretrained=True)
        model.eval()  # Set the model to evaluation mode (no training)

    # Save the model and weights
    save_path = os.path.join(save_directory, f"{model_name}.pth")
    torch.save(model, save_path)

    print(f"Saved {model_name} model and weights to: {save_path}")
