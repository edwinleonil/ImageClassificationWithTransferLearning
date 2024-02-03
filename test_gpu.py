import torch

if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")