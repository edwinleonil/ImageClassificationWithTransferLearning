import torch
import timm
import os

# Define the models to download
models_to_download = {
    "vit_base_patch16_224": timm.models.vision_transformer.vit_base_patch16_224,  # ViT from timm library https://timm.fast.ai/
}

# Directory to save the models
save_directory = "pretrained_models"

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Download and save the models
for model_name, model_fn in models_to_download.items():
    model = model_fn(pretrained=True)
    model.eval()  # Set the model to evaluation mode (no training)

    # Save the model and weights
    save_path = os.path.join(save_directory, f"{model_name}.pth")
    torch.save(model, save_path)

    print(f"Saved {model_name} model and weights to: {save_path}")